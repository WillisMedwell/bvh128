#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <xmmintrin.h>

#define USE_VARIANCE true

namespace bvh128::details {

struct aabb {
    __m128 min;
    __m128 max;
};

template <typename T>
concept Array3 = requires(const T t) {
    {
        t[0]
    } -> std::same_as<const float&>;
    {
        t[1]
    } -> std::same_as<const float&>;
    {
        t[2]
    } -> std::same_as<const float&>;
};
auto vectorise(const Array3 auto& min, const Array3 auto& max) noexcept
{
    float min_buffer[4] = { min[0], min[1], min[2], 0.0f };
    float max_buffer[4] = { max[0], max[1], max[2], 0.0f };
    return aabb {
        .min = _mm_loadu_ps(min_buffer),
        .max = _mm_loadu_ps(max_buffer),
    };
}
auto vectorise(const std::array<float, 3> min, const std::array<float, 3> max) noexcept
{
    float min_buffer[4] = { min[0], min[1], min[2], 0.0f };
    float max_buffer[4] = { max[0], max[1], max[2], 0.0f };
    return aabb {
        .min = _mm_loadu_ps(min_buffer),
        .max = _mm_loadu_ps(max_buffer),
    };
}
auto vectorise(const std::array<float, 3> min, const std::array<float, 3> max, uint64_t data) noexcept
{
    const auto data_lower_32 = std::bit_cast<float>(static_cast<uint32_t>(data & 0xFFFFFFFF));
    const auto data_upper_32 = std::bit_cast<float>(static_cast<uint32_t>((data >> 32) & 0xFFFFFFFF));

    float min_buffer[4] = { min[0], min[1], min[2], data_lower_32 };
    float max_buffer[4] = { max[0], max[1], max[2], data_upper_32 };

    return aabb {
        .min = _mm_loadu_ps(min_buffer),
        .max = _mm_loadu_ps(max_buffer),
    };
}

auto devectorise(const aabb& a) noexcept
{
    float min_buffer[4];
    float max_buffer[4];

    _mm_store_ps(min_buffer, a.min);
    _mm_store_ps(max_buffer, a.max);

    auto data_lower_32 = static_cast<uint64_t>(std::bit_cast<uint32_t>(min_buffer[3]));
    auto data_upper_32 = static_cast<uint64_t>(std::bit_cast<uint32_t>(max_buffer[3])) << 32;
    auto data = data_lower_32 | data_upper_32;

    return std::tuple {
        std::array { min_buffer[0], min_buffer[1], min_buffer[2] },
        std::array { max_buffer[0], max_buffer[1], max_buffer[2] },
        data,
    };
}

auto does_intersect(const details::aabb& lhs, const details::aabb& rhs) noexcept
{
    auto lhs_greater = _mm_cmple_ps(lhs.min, rhs.max);
    auto rhs_greater = _mm_cmple_ps(rhs.min, lhs.max);
    auto is_within = _mm_and_ps(lhs_greater, rhs_greater);
    auto mask = _mm_movemask_ps(is_within);
    return (mask & 0b0111) == 0b0111;
}

auto does_intersect(const details::aabb& lhs, const __m128& rhs) noexcept
{
    auto lhs_greater = _mm_cmple_ps(lhs.min, rhs);
    auto rhs_greater = _mm_cmple_ps(rhs, lhs.max);
    auto is_within = _mm_and_ps(lhs_greater, rhs_greater);
    auto mask = _mm_movemask_ps(is_within);
    return (mask & 0b0111) == 0b0111;
}

auto has_equal_minmax(const details::aabb& lhs, const details::aabb& rhs) noexcept
{
    auto min_eq = _mm_cmplt_ps(lhs.min, rhs.max);
    auto max_eq = _mm_cmplt_ps(rhs.min, lhs.max);
    auto both_eq = _mm_and_ps(min_eq, max_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return (mask & 0b0111) == 0b0111;
}

auto has_equal_minmax_data(const details::aabb& lhs,
    const details::aabb& rhs) noexcept
{
    auto min_eq = _mm_cmplt_ps(lhs.min, rhs.max);
    auto max_eq = _mm_cmplt_ps(rhs.min, lhs.max);
    auto both_eq = _mm_and_ps(min_eq, max_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return mask == 0b1111;
}

auto calc_middle(const details::aabb& a) noexcept
{
    const auto diff_halved = _mm_div_ps(_mm_sub_ps(a.max, a.min), _mm_set_ps1(2.0f));
    return _mm_add_ps(a.min, diff_halved);
}

auto calc_bounding_volume_mean(const details::aabb* aabb_begin, uint32_t aabb_sz)
{
    const auto aabb_end = aabb_begin + aabb_sz;

    // calc bounding volume and mean
    details::aabb bounding_volume = *aabb_begin;
    auto mean = calc_middle(*aabb_begin);

    for (auto iter = aabb_begin + 1; iter != aabb_end; ++iter) {
        bounding_volume.min = _mm_min_ps(bounding_volume.min, iter->min);
        bounding_volume.max = _mm_max_ps(bounding_volume.max, iter->max);
        mean = _mm_add_ps(mean, calc_middle(*iter));
    }
    mean = _mm_div_ps(mean, _mm_set_ps1(static_cast<float>(aabb_sz)));

    return std::tuple { bounding_volume, mean };
}

auto calc_bounding_volume_mean_variance(const details::aabb* aabb_begin,
    uint32_t aabb_sz)
{
    const auto aabb_end = aabb_begin + aabb_sz;
    // calc bounding volume and mean
    details::aabb bounding_volume = *aabb_begin;
    auto mean = calc_middle(*aabb_begin);

    for (auto iter = aabb_begin + 1; iter != aabb_end; ++iter) {
        bounding_volume.min = _mm_min_ps(bounding_volume.min, iter->min);
        bounding_volume.max = _mm_max_ps(bounding_volume.max, iter->max);
        mean = _mm_add_ps(mean, calc_middle(*iter));
    }
    mean = _mm_div_ps(mean, _mm_set_ps1(static_cast<float>(aabb_sz)));

    // calc variance
    auto variance = _mm_setzero_ps();
    for (auto iter = aabb_begin; iter != aabb_end; ++iter) {
        auto diff = _mm_sub_ps(mean, calc_middle(*iter));
        auto squared_diff = _mm_mul_ps(diff, diff);
        variance = _mm_add_ps(variance, squared_diff);
    }
    // (not dividing variance at the end, as no point; its just a scalar anyway)
    return std::tuple { bounding_volume, mean, variance };
}

auto branchless_minmidmax_indices(const float* vec3) noexcept
{
    const unsigned int a = static_cast<unsigned int>(vec3[0] > vec3[1]);
    const unsigned int b = static_cast<unsigned int>(vec3[1] > vec3[2]) << 1;
    const unsigned int c = static_cast<unsigned int>(vec3[2] > vec3[0]) << 2;
    const auto index = (a | b | c);

    using A3 = std::array<int, 3>;
    // Invalid                        0b 0 0 0 = 0
    constexpr A3 bca = { 1, 2, 0 }; // 0b 0 0 1 = 1
    constexpr A3 cab = { 2, 0, 1 }; // 0b 0 1 0 = 2
    constexpr A3 cba = { 2, 1, 0 }; // 0b 0 1 1 = 3
    constexpr A3 abc = { 0, 1, 2 }; // 0b 1 0 0 = 4
    constexpr A3 bac = { 1, 0, 2 }; // 0b 1 0 1 = 5
    constexpr A3 acb = { 0, 2, 1 }; // 0b 1 1 0 = 6
    //                                0b 1 1 1 = 7

    constexpr static A3 lookup[8] = { abc, bca, cab, cba, abc, bac, acb, abc };

    return lookup[index];
}

auto replace_value_at_index(const __m128& dest, const __m128& src,
    const size_t index) noexcept
{
    auto result = dest;
    auto src_data = reinterpret_cast<const float*>(&src);
    auto data = reinterpret_cast<float*>(&result);
    data[index] = src_data[index];
    return result;
}

auto split(details::aabb* aabb_begin, uint32_t aabb_sz) noexcept
{
    // 2n operation
#if USE_VARIANCE
    auto [bv, mean, variance] = calc_bounding_volume_mean_variance(aabb_begin, aabb_sz);
    auto [min_i, mid_i, max_i] = branchless_minmidmax_indices(reinterpret_cast<const float*>(&variance));

#else
    auto [bv, mean] = calc_bounding_volume_mean(aabb_begin, aabb_sz);

    auto length = _mm_sub_ps(bv.max, bv.min);
    auto [min_i, mid_i, max_i] = branchless_minmidmax_indices(reinterpret_cast<const float*>(&length));
#endif

    auto partition_volume = details::aabb {
        bv.min,
        replace_value_at_index(bv.max, mean, max_i),
    };

    auto aabb_end = aabb_begin + aabb_sz;
    auto aabb_mid = std::partition(aabb_begin, aabb_end, [&](const details::aabb& aabb) {
        return does_intersect(partition_volume, calc_middle(aabb));
    });

    details::aabb* adjusted_aabb_mid[2] = {
        aabb_mid,
        aabb_mid - 1,
    };
    aabb_mid = adjusted_aabb_mid[aabb_mid == aabb_end];

    return std::tuple { bv, aabb_mid };
}

} // namespace bvh::simd

namespace bvh128 {

class tree {
    struct unexpanded {
        bvh128::details::aabb* aabbs_begin;
        uint32_t aabbs_sz;
    };
    struct branch {
        uint32_t first;
    };
    union stored {
        unexpanded u;
        branch b;
    };
    enum class stored_id {
        unexpanded = 0,
        branch = 1,
        leaf = 2,
    };
    struct node {
        bvh128::details::aabb aabb;
        stored s;
        stored_id s_id;
    };
    struct M {
        std::unique_ptr<uint32_t[]> iteration_buffer;
        std::unique_ptr<bvh128::details::aabb[]> aabbs;
        std::unique_ptr<node[]> nodes;
        uint32_t nodes_sz;
        uint32_t aabbs_sz;
        bool is_expanded;
    } _m;

    tree() = delete;
    explicit tree(M&& m)
        : _m(std::move(m))
    {
    }

    auto expand(node& node, struct node* nodes)
    {
        if (node.s.u.aabbs_sz == 1) {
            node.aabb = *node.s.u.aabbs_begin;
            node.s_id = stored_id::leaf;
            return;
        }
        auto [aabb, aabb_mid] = split(node.s.u.aabbs_begin, node.s.u.aabbs_sz);

        uint32_t lhs_aabbs_sz = aabb_mid - node.s.u.aabbs_begin;
        uint32_t rhs_aabbs_sz = node.s.u.aabbs_sz - lhs_aabbs_sz;

        auto lhs_index = _m.nodes_sz++;
        auto& lhs = nodes[lhs_index];
        lhs.s.u = { node.s.u.aabbs_begin, lhs_aabbs_sz };
        lhs.s_id = stored_id::unexpanded;

        auto rhs_index = _m.nodes_sz++;
        auto& rhs = nodes[rhs_index];
        rhs.s.u = { aabb_mid, rhs_aabbs_sz };
        rhs.s_id = stored_id::unexpanded;

        node.aabb = aabb;
        node.s.b = branch { lhs_index };
        node.s_id = stored_id::branch;
    }

public:
    static auto create(std::unique_ptr<bvh128::details::aabb[]>&& aabbs,
        uint32_t aabbs_sz)
    {
        const size_t max_nodes = static_cast<size_t>(aabbs_sz) * 2 - 1;
        auto nodes = std::make_unique_for_overwrite<node[]>(max_nodes);
        auto iteration_buffer = std::make_unique_for_overwrite<uint32_t[]>(max_nodes);

        node& front_node = *nodes.get();
        front_node.s.u = unexpanded { aabbs.get(), aabbs_sz };
        front_node.s_id = stored_id::unexpanded;

        return tree(M {
            .iteration_buffer = std::move(iteration_buffer),
            .aabbs = std::move(aabbs),
            .nodes = std::move(nodes),
            .nodes_sz = 1,
            .aabbs_sz = aabbs_sz,
            .is_expanded = false,
        });
    }

    template <typename Pred>
    auto foreach_intersection(bvh128::details::aabb aabb, Pred pred)
    {
        uint32_t* queue = _m.iteration_buffer.get();
        queue[0] = 0;
        uint32_t queue_sz = 1;

        auto nodes = _m.nodes.get();

        for (uint32_t i = 0; i < queue_sz; ++i) {
            node& node = nodes[queue[i]];

            if (node.s_id == stored_id::unexpanded) {
                expand(node, nodes);
            }
            if (details::does_intersect(node.aabb, aabb)) {
                if (node.s_id == stored_id::branch) {
                    queue[queue_sz++] = node.s.b.first;
                    queue[queue_sz++] = node.s.b.first + 1;
                } else {
                    pred(node.aabb);
                }
            }
        }
    }

    template <typename Pred>
    auto foreach_intersection_linear(bvh128::details::aabb aabb, Pred pred)
    {
        auto begin = _m.aabbs.get();
        auto end = begin + _m.aabbs_sz;

        for (auto iter = begin; iter != end; ++iter) {
            if (does_intersect(aabb, *iter)) {
                pred(*iter);
            }
        }
    }
};
} // namespace bvh