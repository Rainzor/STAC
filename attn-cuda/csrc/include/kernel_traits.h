#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace stac_attn {

using namespace cute;

template <typename Element_, int kBlockM_ = 128, int kBlockN_ = 64,
          int kHeadDim_ = 64, int kNWarps_ = 4, int kStages_ = 2>
struct FwdKernelTraits {
    using Element = Element_;
    using ElementAccum = float;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kStages = kStages_;
    static constexpr int kNumThreads = kNWarps * 32;

    // SM80 tensor core MMA atom: fp16 -> fp32 accumulator
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<Element, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

    // Tile the MMA across warps: kNWarps warps along M
    // Each warp handles 16 rows of M, so kNWarps warps -> 16*kNWarps M rows
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0);

    // Memory layout: each row = kHeadDim elements = kHeadDim * 2 bytes (fp16)
    // For D=64, fp16: 128 bytes per row = 1 cache line
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem =
        (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

    // Swizzle parameters to avoid bank conflicts
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4
                                  : (kBlockKGmem == 64 ? 3
                                  : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2
                                      : (sizeof(Element) == 2 ? 3 : 4);

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>, Int<kStages>>{}));

    // Transposed V layout for P @ V^T GEMM
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(
                        Shape<Int<kHeadDim>, Int<kBlockN>, Int<kStages>>{},
                        Step<_2, _1, _3>{})));

    // Shared memory copy atoms for smem -> register
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

    // Global memory copy atom: 128-bit cp.async
    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    // Thread layout for global memory loads
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(kNumThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom = Layout<
        Shape<Int<kNumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(GmemCopyAtom{}, GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

    // Single-stage KV layout
    using SmemLayoutKV1 = decltype(tile_to_shape(SmemLayoutAtom{},
                                                  Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // Q (persistent) + K (2-stage double-buffer) + V (2-stage double-buffer)
    static constexpr int kSmemQSize = int(cosize(SmemLayoutQ{})) * sizeof(Element);
    static constexpr int kSmemKSize = int(cosize(SmemLayoutK{})) * sizeof(Element);
    static constexpr int kSmemVSize = int(cosize(SmemLayoutV{})) * sizeof(Element);
    static constexpr int kSmemBiasSize = kBlockN * sizeof(float);
    static constexpr int kSmemSize = kSmemQSize + kSmemKSize + kSmemVSize + kSmemBiasSize;
};

}  // namespace stac_attn
