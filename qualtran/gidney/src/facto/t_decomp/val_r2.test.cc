#include "val_r2.h"

#include "gtest/gtest.h"

using namespace decomp;

TEST(valr2, add) {
    ValR2 a{.r=2, .r2=3, .divisions_by_2=5};
    ValR2 b{.r=7, .r2=11, .divisions_by_2=2};
    ASSERT_EQ(a + b, b + a);
    ASSERT_NEAR((a + b).approx(), a.approx() + b.approx(), 1e-8);
}

TEST(valr2, sub) {
    ValR2 a{.r=2, .r2=3, .divisions_by_2=5};
    ValR2 b{.r=7, .r2=11, .divisions_by_2=2};
    ASSERT_EQ(a - b, -(b - a));
    ASSERT_NEAR((b - a).approx(), b.approx() - a.approx(), 1e-8);
}

TEST(valr2, mul) {
    ValR2 a{.r=2, .r2=3, .divisions_by_2=5};
    ValR2 b{.r=7, .r2=11, .divisions_by_2=2};
    ASSERT_EQ(a * b, b * a);
    ASSERT_NEAR((b * a).approx(), b.approx() * a.approx(), 1e-8);
}

