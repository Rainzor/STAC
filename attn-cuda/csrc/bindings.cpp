#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <string>

#define ATTN_CUDA_VERSION_MAJOR 0
#define ATTN_CUDA_VERSION_MINOR 1
#define ATTN_CUDA_VERSION_PATCH 0

namespace stac_attn {

void stac_flash_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor out, torch::Tensor lse,
    float softmax_scale,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> colsum,
    int seqlen_q_sub,
    int q_m_stride);

}  // namespace stac_attn

bool is_available() { return true; }

std::string get_version() {
    return std::to_string(ATTN_CUDA_VERSION_MAJOR) + "." +
           std::to_string(ATTN_CUDA_VERSION_MINOR) + "." +
           std::to_string(ATTN_CUDA_VERSION_PATCH);
}

std::vector<torch::Tensor> flash_attn_fwd(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    float softmax_scale,
    c10::optional<torch::Tensor> bias,
    bool return_colsum,
    float subsample_ratio)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.dtype() == torch::kHalf || q.dtype() == torch::kBFloat16,
                "q must be fp16 or bf16");
    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, M, H, D]");
    TORCH_CHECK(k.dim() == 4, "k must be 4D [B, N, H, D]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [B, N, H, D]");

    const int B = q.size(0);
    const int M = q.size(1);
    const int H = q.size(2);
    const int D = q.size(3);
    const int N = k.size(1);

    TORCH_CHECK(D == 64, "Only head_dim=64 is supported, got ", D);
    TORCH_CHECK(k.size(0) == B && v.size(0) == B, "Batch size mismatch");
    TORCH_CHECK(k.size(2) == H && v.size(2) == H, "Head count mismatch");
    TORCH_CHECK(k.size(3) == D && v.size(3) == D, "Head dim mismatch");
    TORCH_CHECK(subsample_ratio > 0.0f && subsample_ratio <= 1.0f,
                "subsample_ratio must be in (0, 1], got ", subsample_ratio);

    const int BLOCK_M = 128;
    int M_sub;
    int q_m_stride;
    float correction;
    if (subsample_ratio >= 0.999999f) {
        // Keep full-M colsum exact when ratio=1.0.
        M_sub = M;
        q_m_stride = 1;
        correction = 1.0f;
    } else {
        M_sub = std::max(BLOCK_M, static_cast<int>(M * subsample_ratio));
        M_sub = (M_sub / BLOCK_M) * BLOCK_M;
        if (M_sub >= M) {
            M_sub = M;
            q_m_stride = 1;
            correction = 1.0f;
        } else {
            q_m_stride = std::max(1, M / M_sub);
            correction = static_cast<float>(M) / static_cast<float>(M_sub);
        }
    }

    auto opts = q.options();
    torch::Tensor out = torch::empty({B, M, H, D}, opts);
    torch::Tensor lse = torch::empty({B, H, M}, opts.dtype(torch::kFloat32));

    // Normalize bias to contiguous fp32 [B, H, N]
    c10::optional<torch::Tensor> bias_fp32;
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        if (b.dim() == 4) {
            TORCH_CHECK(b.size(2) == 1, "4D bias must have shape [B, H, 1, N], got dim2=", b.size(2));
            b = b.squeeze(2);
        }
        TORCH_CHECK(b.dim() == 3, "bias must be 3D [B, H, N] (or 4D [B, H, 1, N])");
        TORCH_CHECK(b.size(0) == B || b.size(0) == 1, "bias batch must be B or 1");
        TORCH_CHECK(b.size(1) == H, "bias heads must match, got ", b.size(1));
        TORCH_CHECK(b.size(2) == N, "bias N must match K seqlen, got ", b.size(2));
        if (b.size(0) == 1 && B > 1) {
            b = b.expand({B, H, N});
        }
        bias_fp32 = b.to(torch::kFloat32).contiguous();
    }

    c10::optional<torch::Tensor> colsum_opt;
    if (return_colsum) {
        colsum_opt = torch::zeros({B, H, N}, opts.dtype(torch::kFloat32));
    }

    stac_attn::stac_flash_fwd(q, k, v, out, lse, softmax_scale, bias_fp32, colsum_opt,
                               M_sub, q_m_stride);

    if (return_colsum && correction != 1.0f) {
        colsum_opt.value().mul_(correction);
    }

    std::vector<torch::Tensor> result = {out, lse};
    if (return_colsum) {
        result.push_back(colsum_opt.value());
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("is_available", &is_available, "Check if attn_cuda is available");
    m.def("get_version", &get_version, "Get version string");
    m.def("flash_attn_fwd", &flash_attn_fwd,
          "Flash attention forward with optional bias and colsum",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("softmax_scale"),
          py::arg("bias") = py::none(),
          py::arg("return_colsum") = false,
          py::arg("subsample_ratio") = 1.0f);
}
