#include "val.h"

#include "gtest/gtest.h"

using namespace decomp;

static void EXPECT_NEAR_COMPLEX(std::complex<double> a, std::complex<double> b, double atol) {
    auto c = a - b;
    auto r = c.real();
    auto i = c.imag();
    auto d = sqrt(r*r + i*i);
    EXPECT_TRUE(d <= atol) << a << " vs " << b << " (" << d << " not within " << atol << ")";
}

TEST(val, eq) {
    Val v{.r=1, .i=2, .r2=3, .i2=4};
    ASSERT_TRUE(v == (Val{.r=1, .i=2, .r2=3, .i2=4}));
    ASSERT_FALSE(v == (Val{.r=9, .i=2, .r2=3, .i2=4}));
    ASSERT_FALSE(v == (Val{.r=1, .i=9, .r2=3, .i2=4}));
    ASSERT_FALSE(v == (Val{.r=1, .i=2, .r2=9, .i2=4}));
    ASSERT_FALSE(v == (Val{.r=1, .i=2, .r2=3, .i2=9}));
    ASSERT_FALSE(v != (Val{.r=1, .i=2, .r2=3, .i2=4}));
    ASSERT_TRUE(v != (Val{.r=9, .i=2, .r2=3, .i2=4}));
    ASSERT_TRUE(v != (Val{.r=1, .i=9, .r2=3, .i2=4}));
    ASSERT_TRUE(v != (Val{.r=1, .i=2, .r2=9, .i2=4}));
    ASSERT_TRUE(v != (Val{.r=1, .i=2, .r2=3, .i2=9}));
    ASSERT_EQ(v.str(), "1+2i+3√2+4i√2");
    EXPECT_NEAR_COMPLEX(v.approx(), (std::complex<double>{5.24264068712, 7.65685424949}), 1e-10);
}

TEST(val, add) {
    Val v1{.r=1, .i=2, .r2=3, .i2=4};
    Val v2{.r=10, .i=20, .r2=30, .i2=40};
    Val v3{.r=11, .i=22, .r2=33, .i2=44};
    ASSERT_EQ(v1 + v2, v3);

    Val t1{.r=2, .i=5, .r2=11, .i2=17};
    Val t2{.r=3, .i=7, .r2=13, .i2=19};
    EXPECT_NEAR_COMPLEX((t1 + t2).approx(), t1.approx() + t2.approx(), 1e-10);
}

TEST(val, left_shift) {
    Val v1{.r=1, .i=2, .r2=3, .i2=4};
    Val v2{.r=4, .i=8, .r2=12, .i2=16};
    ASSERT_EQ(v1 << 0, v1);
    ASSERT_EQ(v1 << 2, v2);

    Val t1{.r=2, .i=5, .r2=11, .i2=17};
    EXPECT_NEAR_COMPLEX(t1.approx() * std::complex<double>(4, 0), (t1 << 2).approx(), 1e-10);
}

TEST(val, times_sqrt2) {
    Val v1{.r=1, .i=2, .r2=3, .i2=4};
    ASSERT_EQ(v1.times_sqrt2(), (Val{.r=6, .i=8, .r2=1, .i2=2}));
    ASSERT_EQ(v1.times_sqrt2().times_sqrt2(), v1 << 1);

    Val t1{.r=2, .i=5, .r2=11, .i2=17};
    EXPECT_NEAR_COMPLEX(t1.approx() * std::complex<double>(sqrt(2), 0), t1.times_sqrt2().approx(), 1e-10);
}

TEST(val, mul) {
    Val v{.r=1, .i=2, .r2=3, .i2=4};
    ASSERT_EQ(v * (Val{.r=1, .i=0, .r2=0, .i2=0}), (Val{.r=1, .i=2, .r2=3, .i2=4}));
    ASSERT_EQ(v * (Val{.r=0, .i=1, .r2=0, .i2=0}), (Val{.r=-2, .i=1, .r2=-4, .i2=3}));
    ASSERT_EQ(v * (Val{.r=0, .i=0, .r2=1, .i2=0}), (Val{.r=6, .i=8, .r2=1, .i2=2}));
    ASSERT_EQ(v * (Val{.r=0, .i=0, .r2=0, .i2=1}), (Val{.r=-8, .i=6, .r2=-2, .i2=1}));

    Val t1{.r=2, .i=5, .r2=11, .i2=17};
    Val t2{.r=3, .i=7, .r2=13, .i2=19};
    EXPECT_NEAR_COMPLEX((t1 * t2).approx(), t1.approx() * t2.approx(), 1e-10);
}

TEST(val, sqrt2_divisibility) {
    ASSERT_EQ((Val{.r=1, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=2, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 2);
    ASSERT_EQ((Val{.r=3, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=4, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 4);
    ASSERT_EQ((Val{.r=5, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=6, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 2);
    ASSERT_EQ((Val{.r=7, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=8, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 6);
    ASSERT_EQ((Val{.r=9, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 0);

    ASSERT_EQ((Val{.r=0, .i=1, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=0, .i=2, .r2=0, .i2=0}).sqrt2_divisibility(), 2);
    ASSERT_EQ((Val{.r=0, .i=3, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=0, .i=4, .r2=0, .i2=0}).sqrt2_divisibility(), 4);
    ASSERT_EQ((Val{.r=0, .i=5, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=0, .i=6, .r2=0, .i2=0}).sqrt2_divisibility(), 2);
    ASSERT_EQ((Val{.r=0, .i=7, .r2=0, .i2=0}).sqrt2_divisibility(), 0);
    ASSERT_EQ((Val{.r=0, .i=8, .r2=0, .i2=0}).sqrt2_divisibility(), 6);
    ASSERT_EQ((Val{.r=0, .i=9, .r2=0, .i2=0}).sqrt2_divisibility(), 0);

    ASSERT_EQ((Val{.r=0, .i=0, .r2=1, .i2=0}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=2, .i2=0}).sqrt2_divisibility(), 3);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=3, .i2=0}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=4, .i2=0}).sqrt2_divisibility(), 5);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=5, .i2=0}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=6, .i2=0}).sqrt2_divisibility(), 3);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=7, .i2=0}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=8, .i2=0}).sqrt2_divisibility(), 7);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=9, .i2=0}).sqrt2_divisibility(), 1);

    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=1}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=2}).sqrt2_divisibility(), 3);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=3}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=4}).sqrt2_divisibility(), 5);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=5}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=6}).sqrt2_divisibility(), 3);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=7}).sqrt2_divisibility(), 1);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=8}).sqrt2_divisibility(), 7);
    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=9}).sqrt2_divisibility(), 1);

    ASSERT_EQ((Val{.r=8, .i=8, .r2=8, .i2=4}).sqrt2_divisibility(), 5);
    ASSERT_EQ((Val{.r=8, .i=8, .r2=4, .i2=8}).sqrt2_divisibility(), 5);
    ASSERT_EQ((Val{.r=8, .i=4, .r2=8, .i2=8}).sqrt2_divisibility(), 4);
    ASSERT_EQ((Val{.r=4, .i=8, .r2=8, .i2=8}).sqrt2_divisibility(), 4);

    ASSERT_EQ((Val{.r=0, .i=0, .r2=0, .i2=0}).sqrt2_divisibility(), 128);
}
