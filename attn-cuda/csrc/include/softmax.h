#pragma once

#include <cmath>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace stac_attn {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template<>
struct MaxOp<float> {
    __device__ __forceinline__ float operator()(float const& x, float const& y) { return max(x, y); }
};

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(T const& x, T const& y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

template<>
struct Allreduce<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator& op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void thread_reduce_(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1>& summary, Operator& op)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void quad_allreduce_(
    Tensor<Engine0, Layout0>& dst,
    Tensor<Engine1, Layout1>& src, Operator& op)
{
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1, typename Operator>
CUTLASS_DEVICE void reduce_(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1>& summary, Operator& op)
{
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0,
         typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_max(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1>& max_val)
{
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max_val, max_op);
}

template<bool zero_init=true, bool warp_reduce=true,
         typename Engine0, typename Layout0,
         typename Engine1, typename Layout1>
CUTLASS_DEVICE void reduce_sum(
    Tensor<Engine0, Layout0> const& tensor,
    Tensor<Engine1, Layout1>& sum)
{
    SumOp<float> sum_op;
    thread_reduce_<zero_init>(tensor, sum, sum_op);
    if constexpr (warp_reduce) { quad_allreduce_(sum, sum, sum_op); }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert SM80 acc layout (MMA=4, MMA_M, MMA_N) -> (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout0>
CUTLASS_HOST_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)),
                       make_layout(get<0, 0>(l), get<2>(l)));
}

// Convert acc layout for A-register retiling: (MMA=4, MMA_M, MMA_N) -> ((4,2), MMA_M, MMA_N/2)
template<typename TiledMma, typename Layout0>
CUTLASS_HOST_DEVICE auto convert_layout_acc_Aregs(Layout0 acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename TiledMma::AtomShape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N/2))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Scale_max=true, bool Check_inf=true,
         typename Engine0, typename Layout0,
         typename Engine1, typename Layout1>
CUTLASS_DEVICE void scale_apply_exp2(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1> const& max_val, float const scale)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max_val) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float const max_scaled = Check_inf
            ? (max_val(mi) == -INFINITY ? 0.f : (!Scale_max ? max_val(mi) : max_val(mi) * scale))
            : (!Scale_max ? max_val(mi) : max_val(mi) * scale);
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kNRows>
struct Softmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    CUTLASS_DEVICE Softmax(float softmax_scale_log2_)
        : softmax_scale_log2(softmax_scale_log2_) {}

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    CUTLASS_DEVICE TensorT max_get_scale(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        TensorT scores_scale;
        if constexpr (Is_first) {
            reduce_max<true>(scores, row_max);
            cute::fill(scores_scale, 1.f);
        } else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            reduce_max<false>(scores, row_max);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    }

    template<bool Is_first, bool Check_inf=false, typename Tensor0>
    CUTLASS_DEVICE void online_softmax(Tensor0& acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(CUTE_STATIC_V(size<0>(scores)) == kNRows);
        scale_apply_exp2<true, Check_inf>(scores, row_max, softmax_scale_log2);
        reduce_sum<Is_first, false>(scores, row_sum);
    }

    CUTLASS_DEVICE TensorT finalize() {
        SumOp<float> sum_op;
        quad_allreduce_(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum;
            row_sum(mi) = (sum == 0.f || sum != sum)
                ? -INFINITY
                : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    }

    template<typename Tensor1>
    CUTLASS_DEVICE void rescale_o(Tensor1& acc_o, TensorT const& scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        static_assert(CUTE_STATIC_V(size<0>(acc_o_rowcol)) == kNRows);
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale(mi);
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int N>
CUTLASS_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace stac_attn
