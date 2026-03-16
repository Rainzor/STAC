# attn-cuda 性能优化方案

## 目标配置

```
M = 1024 × 4 = 4096   (query tokens, ~4 frames × chunk)
N = 1024 × 16 = 16384  (key tokens, ~16 frames cache)
H = 16                  (attention heads)
D = 64                  (head dimension)
B = 1                   (batch size)
dtype = fp16/bf16
GPU = SM80+ (RTX 3090 / A100)
```

## 现状（已实现 Phase A + C 之后）

Bench 结果 (RTX 3090, fp16)，见 `attn-cuda/tests/compare_cuda_torch_triton.py`：

**纯 O+LSE（无 ColSum）**:

| Config | SDPA (ms) | attn_cuda (ms) | 说明 |
|--------|-----------|----------------|------|
| M=4096 N=16384 | ~4.4 | ~6.1 | CUDA 约慢 0.71x（SDPA 更快） |
| M=1024 N=16384 | ~1.2 | ~1.8 | 同上 |

**Bias + ColSum（Pass1 + Pass2，O+LSE+ColSum）**:

| Config | Triton (ms) | attn_cuda (ms) | Tri/CUDA |
|--------|-------------|----------------|----------|
| M=4096 N=16384 no_bias | ~10.7 | ~11.7 | 0.92x |
| M=4096 N=16384 bias_inf | ~11.8 | ~14.1 | 0.84x |
| M=4096 N=16384 bias_random | ~11.2 | ~14.4 | **0.78x** |

结论：在目标配置下 **CUDA 仍比 Triton 慢约 22%**（bias_random 时 Triton 更快）。Phase A（ColSum 拆成 N-major）和 Phase C（kBlockN=128）已落地，差距已从早期的 ~2.3x 收窄到 ~1.2–1.3x，但仍有优化空间。

---

## 当前代码审查与瓶颈（为何仍慢于 Triton）

以下对照 `attn-cuda/csrc` 实际实现，给出**不改代码**下的审查结论与修改方向。

### 1. Pass1 (fwd_kernel.h)：V 加载与计算串行，无重叠

**位置**: `kernels/fwd_kernel.h` 约 L333–338。

**现状**:
- K 已双缓冲：Prologue 预取第一块 K，主循环中 `cp_async_wait<0>()` 后算 S=Q@K^T，同时 `load_k_tile(next_n_block, next_stage)` 预取下一块 K（L270–266），计算与 K 预取已重叠。
- V 仍为单缓冲：在 softmax 与 P 转换之后，`__syncthreads()` → `load_v_tile(cur_n_block)` → `cp_async_fence()` → `cp_async_wait<0>()` → `__syncthreads()` → O+=P@V^T。即**等当前 N-block 的 S 全部算完才加载 V，再等 V 加载完才算 O**，V 的 DRAM 延迟完全暴露，无法与 S 或下一块 K 的预取重叠。

**与 Triton/FA2 的差异**：FA2 通常对 V 也做双缓冲，在算当前块 O 时预取下一块 V，从而隐藏 V 的访存延迟。

**修改建议**:
- 为 V 增加第二块 smem（与 K 类似的双缓冲），或复用「下一块 K」的 smem 槽位（当前块算 O 时下一块 K 已在另一 stage）。
- 主循环结构改为：在算完 S/softmax/P 后，先发起**下一块 V 的 cp_async**，再 `cp_async_wait` 当前块 V，算 O+=P@V^T；下一轮迭代时当前块的 V 已就绪。预期收益约 10–20%（Pass1 占比约一半总耗时时）。

---

### 2. Pass1 (fwd_kernel.h)：Bias 按元素读 gmem，重复约 128 倍

**位置**: `kernels/fwd_kernel.h` L284–296。

**现状**:
```cpp
if constexpr (HasBias) {
    float const* bias_ptr = params.bias_ptr + ...;
    float const inv_scale = 1.0f / params.softmax_scale;
    int const n_offset = cur_n_block * kBlockN;
    #pragma unroll
    for (int i = 0; i < size(tSrS); ++i) {
        int n_idx = get<1>(tScS(i)) + n_offset;
        if (n_idx < seqlen_k) {
            tSrS(i) += bias_ptr[n_idx] * inv_scale;
        }
    }
}
```
- 每个 N-block 内，scores 张量 `tSrS` 的 shape 为 `[kBlockM, kBlockN]`（如 128×128），即 128 个不同的 `n_idx`，但每个 `n_idx` 在循环中被重复访问约 **kBlockM=128 次**（不同 m 的 fragment 共享同一 n）。
- 因此每个 N-block 对 `bias_ptr` 的 gmem 读取约 **128×128 = 16384 次**，而实际不同的 bias 值只有 **128 个**，相当于每值从 gmem 读约 128 次。

**修改建议**:
- 每个 N-block 开始时，用一次**合并读取**将 `bias[n_offset : n_offset + kBlockN]` 写入 smem（或寄存器数组），再在循环中从 smem 按 `n_idx - n_offset` 取值做 `tSrS(i) += bias_smem[...] * inv_scale`。
- 这样每个 N-block 对 bias 的 gmem 访问从 16384 次降为 128 次，且更易合并。预期收益在 bias 路径下约 5–15%（视带宽与 cache 情况）。

---

### 3. Pass2 (colsum_kernel.h)：Block 内用 atomicAdd 做列归约

**位置**: `kernels/colsum_kernel.h` L246–255。

**现状**:
- 每个 CTA 负责一段 N 列（kBlockN=128），沿 M 扫描；在 MMA 的 partition 下，每个线程持有部分 (M, N) 的 scores，在标量循环中算 P=exp(S*scale+bias-LSE) 并累加到 `col_acc[j]`（L212–230）。
- 同一 N 列上的部分和分布在多个线程（不同 warp/行组），因此最后用 `atomicAdd(&colsum_smem[n_local], col_acc[j])` 把各线程的 `col_acc` 归约到 smem（L248–254），再写 gmem（L263–266）。
- 即：**跨线程的列归约依赖 block 内 atomicAdd**，存在竞争与序列化。

**与 Triton 的差异**：Triton 的 `_colsum_n_major_kernel` 中 `colsum += tl.sum(p, axis=0)` 得到的是整块 [BLOCK_N] 的向量（按线程分布），最后 `tl.store(c_ptrs, colsum)` 一次写回，**无 block 内 atomics**。

**修改建议**:
- 在保持「每个 CTA 负责一段 N、扫描 M」的前提下，改为显式按列归约：
  - 方案 A：每个线程先按 MMA 的 (m,n) 布局算出自己对各 n 的贡献，再对每个 n 做 **block 内 reduce**（如 warp shuffle + 一个 lane 写 smem，或 block reduce），最后每个 n 只由一名线程写 `colsum_smem[n]`，再写 gmem。
  - 方案 B：调整 partition，使「每列 n 对应一组线程」的划分更规整，用一次 block-level reduce（如 cub::BlockReduce）按列归约，再写 smem/gmem，完全去掉 atomicAdd。
- 预期收益：Pass2 约 5–15%，整体约 3–8%。

---

### 4. Pass2 (colsum_kernel.h)：M 循环内 Q 加载与计算未重叠

**位置**: `kernels/colsum_kernel.h` L171–193。

**现状**:
- M 循环中：`load Q[mi]` → `cp_async_fence()` → `cp_async_wait<0>()` → `__syncthreads()` → S=Q@K^T → P/reduce → `__syncthreads()`。
- 即每轮都是「等 Q 加载完 → 算 S 与归约」，**下一块 Q 的加载未与当前块的 S 计算重叠**。

**修改建议**:
- 对 Q 做双缓冲（两块 smem）：在算当前块 S 与 colsum 累加的同时，用 cp_async 预取下一块 Q；下一轮开始时 `cp_async_wait` 即可使用，减少 stall。预期收益约 5–10%（仅 Pass2）。

---

### 5. Pass1：kBlockN 与 Triton 已对齐

**位置**: `launch.cu` L59–76。

**现状**: `seqlen_k >= 2048` 时已使用 `FwdKernelTraits<Element, 128, 128>`（kBlockM=128, kBlockN=128），与 Triton BLOCK_N=128 一致；N=16384 时迭代 128 次。此处无需再改。

---

### 6. Pass1：O 写回已向量化

**位置**: `kernels/fwd_kernel.h` L364–382。

**现状**: O 经 smem 暂存后，按 128-bit (`cute::uint128_t`) 写 gmem，已非标量逐元素写。Phase D 已实现。

---

## 已完成的优化（记录）

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase A** | ColSum 拆成 N-major 独立 kernel（`colsum_n_major_kernel`），Grid (ceil(N/128), B*H)，K 只读一次、无跨 CTA atomics | ✅ 已实现 |
| **Phase C** | N≥2048 时 kBlockN=128，与 Triton 一致 | ✅ 已实现 |
| **Phase D** | O 写回 128-bit 向量化 | ✅ 已实现 |
| **K 双缓冲** | Pass1 主循环 K 两 stage，预取与 S=Q@K^T 重叠 | ✅ 已实现 |

---

## 建议的后续修改（按优先级，仅文档，不改代码）

| 优先级 | 改动 | 预期收益 | 位置 | 说明 |
|--------|------|----------|------|------|
| 1 | Pass1：Bias 按 N-block 加载到 smem 再索引 | 5–15% (bias 路径) | `fwd_kernel.h` L284–296 | 消除 128× 重复 gmem 读 |
| 2 | Pass1：V 双缓冲，与 O 计算重叠 | 10–20% (Pass1) | `fwd_kernel.h` 主循环、smem 布局 | 隐藏 V 的 DRAM 延迟 |
| 3 | Pass2：去掉 block 内 atomicAdd，改为按列 block reduce | 5–15% (Pass2) | `colsum_kernel.h` L246–255 | 与 Triton 一致，无 atomics |
| 4 | Pass2：M 循环中 Q 双缓冲 | 5–10% (Pass2) | `colsum_kernel.h` M-loop | 预取下一块 Q 与当前 S 重叠 |

建议实施顺序：**1 → 2 → 3 → 4**。1 和 2 主要缩小与 SDPA/Triton 在 Pass1 上的差距；3 和 4 进一步收窄 Pass2 与 Triton 的差距。

---

## 预期最终性能（若完成上述修改）

在目标配置 (M=4096, N=16384, H=16, D=64, fp16, bias_random) 下：

- **Pass1**：Bias 读减少 + V 双缓冲 → 有望接近或略优于当前 SDPA/FA2 的 forward 耗时。
- **Pass2**：无 atomics + Q 重叠 → 与 Triton 的 ColSum 耗时接近或略优。
- **整体**：**O+LSE+ColSum 有望达到与 Triton 持平或略快**（Tri/CUDA ≥ 1.0x）。

---

## 历史：早期 M-major ColSum 瓶颈（已由 Phase A 解决）

早期实现曾在同一 kernel 内用 M-major 做 ColSum（每个 CTA 负责 128 行 Q、扫描全部 N），导致：

- K 被 32 个 CTA 各读 128 次 → 32× 读放大；
- 跨 CTA 对 colsum 的 atomicAdd 竞争严重。

Phase A 改为 N-major 独立 kernel 后，K 每 CTA 只读一次、无跨 CTA atomics，差距已从 ~2.3x 收窄到当前 ~1.2–1.3x。上文 1–4 为在当前架构下进一步逼近 Triton 的修改建议。
