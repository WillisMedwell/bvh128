#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <vector>
#include <xmmintrin.h>

#define USE_VARIANCE true

namespace bvh12 {

struct aabb {
    std::array<float, 3> min;
    std::array<float, 3> max;
    uint64_t data;
};

class alignas(16) aabb128 {
    struct M {
        __m128 min;
        __m128 max;
    } _m;

    explicit aabb128(M&& m) noexcept;
    explicit aabb128(__m128&& min, __m128&& max) noexcept;

public:
    aabb128() noexcept = default;

    static aabb128 construct(const aabb& other) noexcept;
    // static aabb128 construct(const std::array<float, 3>& min, const std::array<float, 3>& max, uint64_t data) noexcept;

    [[nodiscard]] aabb deconstruct() const noexcept;

    [[nodiscard]] bool does_intersect(const aabb128& other) const noexcept;
    [[nodiscard]] bool does_point_intersect(const __m128& point) const noexcept;

    bool operator==(const aabb128&) = delete;
    bool operator==(const aabb128&) const = delete;
    [[nodiscard]] bool is_equal_region(const aabb128& other) const noexcept;
    [[nodiscard]] bool is_equal_region_and_data(const aabb128& other) const noexcept;

    static aabb128 calc_bounding_volume(const aabb128& lhs, const aabb128& rhs) noexcept;

    [[nodiscard]] __m128 calc_middle() const noexcept;
};

template <typename allocator>
class tree;

template <typename allocator>
tree<allocator> construct_tree(std::span<const aabb> aabbs, allocator alloc = allocator());

template <typename allocator = std::allocator<std::byte>>
class tree {
    enum class node_variant : uint32_t {
        unexpanded,
        branch,
        leaf
    };
    struct node {
        aabb128 bounding_volume;
        aabb128* aabbs_begin;
        uint32_t aabbs_sz;
        node_variant node_type;
    };
    struct M {
        allocator alloc;
        std::byte* data;
        size_t data_sz;
        std::span<node> nodes;
        std::span<aabb128> aabbs;
        std::span<uint32_t> iteration_indices_buffer;
    } _m;

    tree() = delete;
    explicit tree(M&& m) noexcept;

public:
    friend tree<allocator> construct_tree<>(std::span<const aabb> aabbs, allocator alloc); // Friend declaration
    
    template <typename Pred>
    void for_each_intersection(aabb aabb, Pred pred) const noexcept;

    ~tree() noexcept;
};

template <typename allocator = std::allocator<std::byte>>
tree<allocator> construct_tree(std::span<const aabb> aabbs, allocator alloc)
{
    static_assert(std::is_same_v<decltype(alloc.allocate(1)), std::byte*>);

    const size_t aabbs_sz = aabbs.size();
    const size_t nodes_sz = aabbs.size() * 2 - 1;
    const size_t buffer_sz = nodes_sz;

    const size_t aabbs_sz_bytes = aabbs_sz * sizeof(aabb128) + alignof(aabb128);
    const size_t nodes_sz_bytes = nodes_sz * sizeof(typename tree<allocator>::node) + alignof(typename tree<allocator>::node);
    const size_t buffer_sz_bytes = buffer_sz * sizeof(uint32_t) + alignof(uint32_t);

    const size_t data_sz = aabbs_sz_bytes + nodes_sz_bytes + buffer_sz_bytes;
    std::byte* data = alloc.allocate(aabbs_sz_bytes + nodes_sz_bytes + buffer_sz_bytes);
    std::byte* data_offset = data;

    auto span_w_proper_alignment = [&data_offset]<typename T>(size_t n) {
        data_offset += (alignof(T) - ((reinterpret_cast<std::uintptr_t>(data_offset) % alignof(T)))) % alignof(T);
        T* begin = reinterpret_cast<T*>(data_offset);
        data_offset += n * sizeof(T);
        return std::span<T>(begin, n);
    };
    auto nodes = span_w_proper_alignment.template operator()<typename tree<allocator>::node>(nodes_sz);
    auto aabbs128 = span_w_proper_alignment.template operator()<aabb128>(aabbs_sz);
    auto buffer = span_w_proper_alignment.template operator()<uint32_t>(buffer_sz);

    std::transform(aabbs.begin(), aabbs.end(), aabbs128.begin(), &aabb128::construct);

    nodes[0] = typename tree<allocator>::node {
        std::reduce(aabbs128.begin(), aabbs128.end(), aabbs128.front(), &aabb128::calc_bounding_volume),
        aabbs128.data(),
        static_cast<uint32_t>(aabbs_sz),
        tree<allocator>::node_variant::unexpanded
    };

    return tree<allocator>(typename tree<allocator>::M {
        .alloc = std::move(alloc),
        .data = data,
        .data_sz = data_sz,
        .nodes = nodes,
        .aabbs = aabbs128,
        .iteration_indices_buffer = buffer,
    });
}

template <typename allocator>
tree<allocator>::tree(M&& m) noexcept
    : _m(std::move(m))
{
}
template <typename allocator>
tree<allocator>::~tree() noexcept
{
    _m.alloc.deallocate(_m.data, _m.data_sz);
}

aabb128::aabb128(M&& m) noexcept
    : _m(std::move(m))
{
}
aabb128::aabb128(__m128&& min, __m128&& max) noexcept
    : _m(M { .min = min, .max = max })
{
}
aabb128 aabb128::construct(const aabb& other) noexcept
{
    const uint32_t data_lower_32 = static_cast<uint32_t>(other.data & 0xFFFFFFFF);
    const uint32_t data_upper_32 = static_cast<uint32_t>((other.data >> 32) & 0xFFFFFFFF);

    alignas(16) auto min_buffer = std::to_array({ other.min[0], other.min[1], other.min[2], std::bit_cast<float>(data_lower_32) });
    alignas(16) auto max_buffer = std::to_array({ other.max[0], other.max[1], other.max[2], std::bit_cast<float>(data_upper_32) });

    return aabb128(M {
        .min = _mm_load_ps(min_buffer.data()),
        .max = _mm_load_ps(max_buffer.data()),
    });
}

aabb aabb128::deconstruct() const noexcept
{
    static_assert(sizeof(*this) == sizeof(std::array<float, 8>));
    const auto& view_of_self = *reinterpret_cast<const std::array<float, 8>*>(this);

    const auto data_lower_32 = static_cast<uint64_t>(std::bit_cast<uint32_t>(view_of_self[3]));
    const auto data_upper_32 = static_cast<uint64_t>(std::bit_cast<uint32_t>(view_of_self[7])) << 32;

    return aabb {
        .min = { view_of_self[0], view_of_self[1], view_of_self[2] },
        .max = { view_of_self[4], view_of_self[5], view_of_self[6] },
        .data = data_lower_32 | data_upper_32,
    };
}

bool aabb128::does_intersect(const aabb128& other) const noexcept
{
    auto this_max_greater = _mm_cmple_ps(other._m.min, _m.max);
    auto other_max_greater = _mm_cmple_ps(_m.min, other._m.max);
    auto both_greater = _mm_and_ps(this_max_greater, other_max_greater);
    auto mask = _mm_movemask_ps(both_greater);
    return (mask & 0b0111) == 0b0111;
}
bool aabb128::does_point_intersect(const __m128& point) const noexcept
{
    auto point_greater_than_min = _mm_cmple_ps(_m.min, point);
    auto max_greater_than_point = _mm_cmple_ps(point, _m.max);
    auto both_greater = _mm_and_ps(point_greater_than_min, max_greater_than_point);
    auto mask = _mm_movemask_ps(both_greater);
    return (mask & 0b0111) == 0b0111;
}

bool aabb128::is_equal_region(const aabb128& other) const noexcept
{
    auto max_eq = _mm_cmpeq_ps(_m.min, other._m.max);
    auto min_eq = _mm_cmpeq_ps(other._m.min, _m.max);
    auto both_eq = _mm_and_ps(max_eq, min_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return (mask & 0b0111) == 0b0111;
}
bool aabb128::is_equal_region_and_data(const aabb128& other) const noexcept
{
    auto max_eq = _mm_cmpeq_ps(_m.min, other._m.max);
    auto min_eq = _mm_cmpeq_ps(other._m.min, _m.max);
    auto both_eq = _mm_and_ps(max_eq, min_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return mask == 0b1111;
}

aabb128 aabb128::calc_bounding_volume(const aabb128& lhs, const aabb128& rhs) noexcept
{
    return aabb128 {
        _mm_min_ps(lhs._m.min, rhs._m.min),
        _mm_max_ps(lhs._m.max, rhs._m.max),
    };
}
__m128 aabb128::calc_middle() const noexcept
{
    const auto difference_halved = _mm_div_ps(_mm_sub_ps(_m.max, _m.min), _mm_set_ps1(2.0f));
    return _mm_add_ps(_m.min, difference_halved);
}

}

namespace bvh128::details {

struct alignas(16) aabb {
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
inline auto vectorise(const Array3 auto& min, const Array3 auto& max) noexcept
{
    alignas(16) float min_buffer[4] = { min[0], min[1], min[2], 0.0f };
    alignas(16) float max_buffer[4] = { max[0], max[1], max[2], 0.0f };
    return aabb {
        .min = _mm_load_ps(min_buffer),
        .max = _mm_load_ps(max_buffer),
    };
}
inline auto vectorise(const std::array<float, 3> min, const std::array<float, 3> max) noexcept
{
    alignas(16) float min_buffer[4] = { min[0], min[1], min[2], 0.0f };
    alignas(16) float max_buffer[4] = { max[0], max[1], max[2], 0.0f };
    return aabb {
        .min = _mm_load_ps(min_buffer),
        .max = _mm_load_ps(max_buffer),
    };
}
inline auto vectorise(const std::array<float, 3> min, const std::array<float, 3> max, uint64_t data) noexcept
{
    const auto data_lower_32 = std::bit_cast<float>(static_cast<uint32_t>(data & 0xFFFFFFFF));
    const auto data_upper_32 = std::bit_cast<float>(static_cast<uint32_t>((data >> 32) & 0xFFFFFFFF));

    alignas(16) float min_buffer[4] = { min[0], min[1], min[2], data_lower_32 };
    alignas(16) float max_buffer[4] = { max[0], max[1], max[2], data_upper_32 };

    return aabb {
        .min = _mm_load_ps(min_buffer),
        .max = _mm_load_ps(max_buffer),
    };
}

inline auto devectorise(const aabb& a) noexcept
{
    alignas(16) float min_buffer[4];
    alignas(16) float max_buffer[4];

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

inline auto does_intersect(const details::aabb& lhs, const details::aabb& rhs) noexcept
{
    auto lhs_greater = _mm_cmple_ps(lhs.min, rhs.max);
    auto rhs_greater = _mm_cmple_ps(rhs.min, lhs.max);
    auto is_within = _mm_and_ps(lhs_greater, rhs_greater);
    auto mask = _mm_movemask_ps(is_within);
    return (mask & 0b0111) == 0b0111;
}
inline auto does_intersect(const details::aabb& lhs, const __m128& rhs) noexcept
{
    auto lhs_greater = _mm_cmple_ps(lhs.min, rhs);
    auto rhs_greater = _mm_cmple_ps(rhs, lhs.max);
    auto is_within = _mm_and_ps(lhs_greater, rhs_greater);
    auto mask = _mm_movemask_ps(is_within);
    return (mask & 0b0111) == 0b0111;
}

inline auto has_equal_minmax(const details::aabb& lhs, const details::aabb& rhs) noexcept
{
    auto min_eq = _mm_cmplt_ps(lhs.min, rhs.max);
    auto max_eq = _mm_cmplt_ps(rhs.min, lhs.max);
    auto both_eq = _mm_and_ps(min_eq, max_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return (mask & 0b0111) == 0b0111;
}
inline auto has_equal_minmax_data(const details::aabb& lhs, const details::aabb& rhs) noexcept
{
    auto min_eq = _mm_cmplt_ps(lhs.min, rhs.max);
    auto max_eq = _mm_cmplt_ps(rhs.min, lhs.max);
    auto both_eq = _mm_and_ps(min_eq, max_eq);
    auto mask = _mm_movemask_ps(both_eq);
    return mask == 0b1111;
}

inline auto calc_middle(const details::aabb& a) noexcept
{
    const auto diff_halved = _mm_div_ps(_mm_sub_ps(a.max, a.min), _mm_set_ps1(2.0f));
    return _mm_add_ps(a.min, diff_halved);
}

inline auto calc_bounding_volume_mean(const details::aabb* aabb_begin, uint32_t aabb_sz)
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
inline auto calc_bounding_volume_mean_variance(const details::aabb* aabb_begin, uint32_t aabb_sz)
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

inline auto branchless_minmidmax_indices(const float* vec3) noexcept
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

inline auto replace_value_at_index(const __m128& dest, const __m128& src, const size_t index) noexcept
{
    auto result = dest;
    auto src_data = reinterpret_cast<const float*>(&src);
    auto data = reinterpret_cast<float*>(&result);
    data[index] = src_data[index];
    return result;
}

inline auto split(details::aabb* aabb_begin, uint32_t aabb_sz) noexcept
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

} // namespace bvh128::details

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

    inline auto expand(node& node, struct node* nodes)
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
} // namespace bvh128