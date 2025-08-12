#include "state.h"

#include "gtest/gtest.h"

using namespace decomp;

TEST(ket, x) {
    ASSERT_EQ(Ket::ket_0().after_x(), Ket::ket_1());
    ASSERT_EQ(Ket::ket_1().after_x(), Ket::ket_0());
    ASSERT_EQ(Ket::ket_p().after_x(), Ket::ket_p());
    ASSERT_EQ(Ket::ket_m().after_x(), -Ket::ket_m());
    ASSERT_EQ(Ket::ket_i().after_x(), Ket::ket_j() * (Val{.r=0, .i=1}));
    ASSERT_EQ(Ket::ket_j().after_x(), Ket::ket_i() * (Val{.r=0, .i=-1}));
}

TEST(ket, z) {
    ASSERT_EQ(Ket::ket_0().after_z(), Ket::ket_0());
    ASSERT_EQ(Ket::ket_1().after_z(), -Ket::ket_1());
    ASSERT_EQ(Ket::ket_p().after_z(), Ket::ket_m());
    ASSERT_EQ(Ket::ket_m().after_z(), Ket::ket_p());
    ASSERT_EQ(Ket::ket_i().after_z(), Ket::ket_j());
    ASSERT_EQ(Ket::ket_j().after_z(), Ket::ket_i());
}

TEST(ket, h) {
    ASSERT_EQ(Ket::ket_0().after_h(), Ket::ket_p());
    ASSERT_EQ(Ket::ket_1().after_h(), Ket::ket_m());
    ASSERT_EQ(Ket::ket_p().after_h(), Ket::ket_0());
    ASSERT_EQ(Ket::ket_m().after_h(), Ket::ket_1());
    ASSERT_EQ(Ket::ket_i().after_h(), Ket::ket_j().over2() * (Val{.r2=1, .i2=1}));
    ASSERT_EQ(Ket::ket_j().after_h(), Ket::ket_i().over2() * (Val{.r2=1, .i2=-1}));
}

TEST(ket, h_xy) {
    ASSERT_EQ(Ket::ket_0().after_h_xy(), Ket::ket_1() * (Val{.r=0, .i=1}));
    ASSERT_EQ(Ket::ket_1().after_h_xy(), Ket::ket_0());
    ASSERT_EQ(Ket::ket_p().after_h_xy(), Ket::ket_i());
    ASSERT_EQ(Ket::ket_m().after_h_xy(), -Ket::ket_j());
    ASSERT_EQ(Ket::ket_i().after_h_xy(), Ket::ket_p() * (Val{.r=0, .i=1}));
    ASSERT_EQ(Ket::ket_j().after_h_xy(), Ket::ket_m() * (Val{.r=0, .i=-1}));
}

TEST(ket, h_yz) {
    ASSERT_EQ(Ket::ket_0().after_h_yz(), Ket::ket_i());
    ASSERT_EQ(Ket::ket_1().after_h_yz(), Ket::ket_j() * (Val{.r=0, .i=-1}));
    ASSERT_EQ(Ket::ket_p().after_h_yz(), Ket::ket_m().over2() * (Val{.r2=1, .i2=-1}));
    ASSERT_EQ(Ket::ket_m().after_h_yz(), -Ket::ket_p().over2() * (Val{.r2=-1, .i2=-1}));
    ASSERT_EQ(Ket::ket_i().after_h_yz(), Ket::ket_0());
    ASSERT_EQ(Ket::ket_j().after_h_yz(), Ket::ket_1() * (Val{.r=0, .i=1}));
}

TEST(ket, bloch) {
    ValR2 l{.r=1};
    ValR2 o{.r=0};
    ValR2 s{.r=0, .r2=1, .divisions_by_2=1};
    EXPECT_EQ(Ket::ket_p().bloch_vec(), (std::array<ValR2, 3>{l, o, o}));
    EXPECT_EQ(Ket::ket_m().bloch_vec(), (std::array<ValR2, 3>{-l, o, o}));
    EXPECT_EQ(Ket::ket_i().bloch_vec(), (std::array<ValR2, 3>{o, l, o}));
    EXPECT_EQ(Ket::ket_j().bloch_vec(), (std::array<ValR2, 3>{o, -l, o}));
    EXPECT_EQ(Ket::ket_0().bloch_vec(), (std::array<ValR2, 3>{o, o, l}));
    EXPECT_EQ(Ket::ket_1().bloch_vec(), (std::array<ValR2, 3>{o, o, -l}));
    EXPECT_EQ(Ket::ket_p().after_t().bloch_vec(), (std::array<ValR2, 3>{s, s, o}));
}
