#define CATCH_CONFIG_MAIN

#include <bvh128/bvh128.hpp>
#include <catch2/catch.hpp>
#include <random>

TEST_CASE("Vectorise to aabb", "[aabb]")
{
    auto aabb = bvh128::details::vectorise({ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f });
    REQUIRE(reinterpret_cast<const float*>(&aabb.min)[0] == 0.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb.min)[1] == 1.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb.min)[2] == 2.0f);

    REQUIRE(reinterpret_cast<const float*>(&aabb.max)[0] == 3.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb.max)[1] == 4.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb.max)[2] == 5.0f);

    double data = 123456.789;
    auto aabb_w_data = bvh128::details::vectorise({ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, std::bit_cast<uint64_t>(data));
    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.min)[0] == 0.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.min)[1] == 1.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.min)[2] == 2.0f);

    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.max)[0] == 3.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.max)[1] == 4.0f);
    REQUIRE(reinterpret_cast<const float*>(&aabb_w_data.max)[2] == 5.0f);
}

TEST_CASE("Devectorise from aabb", "[aabb]")
{
    auto aabb = bvh128::details::vectorise({ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f });
    auto [min, max, data] = bvh128::details::devectorise(aabb);
    REQUIRE(min[0] == 0.0f);
    REQUIRE(min[1] == 1.0f);
    REQUIRE(min[2] == 2.0f);

    REQUIRE(max[0] == 3.0f);
    REQUIRE(max[1] == 4.0f);
    REQUIRE(max[2] == 5.0f);

    REQUIRE(data == 0);

    double set_data = 123456.789;
    auto aabb_w_data = bvh128::details::vectorise({ 0.0f, 1.0f, 2.0f }, { 3.0f, 4.0f, 5.0f }, std::bit_cast<uint64_t>(set_data));
    auto [min2, max2, data2] = bvh128::details::devectorise(aabb_w_data);

    REQUIRE(min2[0] == 0.0f);
    REQUIRE(min2[1] == 1.0f);
    REQUIRE(min2[2] == 2.0f);

    REQUIRE(max2[0] == 3.0f);
    REQUIRE(max2[1] == 4.0f);
    REQUIRE(max2[2] == 5.0f);

    REQUIRE(std::bit_cast<double>(data2) == set_data);
}

TEST_CASE("Create a basic tree", "[basic]")
{
    auto gen_rand_aabb = []() {
        static std::random_device rd;
        static std::mt19937 gen(10);
        static std::uniform_real_distribution<> dis_pos(0.0, 100.0);

        static std::uniform_real_distribution<> dis_sz(2.0f, 10.0f);
        std::array<float, 3> min = { static_cast<float>(dis_pos(gen)),
            static_cast<float>(dis_pos(gen)),
            static_cast<float>(dis_pos(gen)) };
        std::array<float, 3> max = { min[0] + static_cast<float>(dis_sz(gen)),
            min[1] + static_cast<float>(dis_sz(gen)),
            min[2] + static_cast<float>(dis_sz(gen)) };

        return bvh128::details::vectorise(min, max);
    };

    size_t aabbs_sz = 100;
    auto aabbs = std::make_unique<bvh128::details::aabb[]>(aabbs_sz);
    for (size_t i = 0; i < aabbs_sz; ++i) {
        aabbs.get()[i] = gen_rand_aabb();
    }
    auto tree = bvh128::tree::create(std::move(aabbs), aabbs_sz);

    { // do a search for a massive aabb.
        size_t count_tree_aabbs = 0;
        tree.foreach_intersection(bvh128::details::vectorise({ 0, 0, 0 }, { 1000, 1000, 1000 }), [&](const auto& aabb) {
            ++count_tree_aabbs;
        });
        REQUIRE(count_tree_aabbs == aabbs_sz);
    }

    { // do a search for a smaller aabb.
        size_t count_tree_search = 0;
        size_t count_linear_search = 0;
        tree.foreach_intersection(bvh128::details::vectorise({ 1, 1, 1 }, { 10, 10, 10 }), [&](const auto& aabb) {
            ++count_tree_search;
        });
        tree.foreach_intersection_linear(bvh128::details::vectorise({ 1, 1, 1 }, { 10, 10, 10 }), [&](const auto& aabb) {
            ++count_linear_search;
        });
        REQUIRE(count_linear_search == count_tree_search);
    }
}
