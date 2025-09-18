#include "mat2.h"

#include "gtest/gtest.h"

using namespace decomp;

static void EXPECT_NEAR_COMPLEX22(
    std::array<std::array<std::complex<double>, 2>, 2> a,
    std::array<std::array<std::complex<double>, 2>, 2> b,
    double atol) {
    double d = 0;
    for (size_t k1 = 0; k1 < 2; k1++) {
        for (size_t k2 = 0; k2 < 2; k2++) {
            auto c = a[k1][k2] - b[k1][k2];
            auto r = c.real();
            auto i = c.imag();
            d += sqrt(r*r + i*i);
        }
    }
    EXPECT_TRUE(d <= atol)
        << a[0][0] << "," << a[0][1] << "," << a[1][0] << "," << a[1][1]
        << "\nvs\n"
        << b[0][0] << "," << b[0][1] << "," << b[1][0] << "," << b[1][1]
        << "\n(" << d << " not within " << atol << ")";
}

TEST(mat2, eq) {
    Val a{.r=1, .i=2, .r2=3, .i2=4};
    Val b{.r=5, .i=8, .r2=11, .i2=14};
    Val c{.r=6, .i=9, .r2=12, .i2=15};
    Val d{.r=7, .i=10, .r2=13, .i2=16};
    Val zero{.r=0, .i=0, .r2=0, .i2=0};
    Mat2 m{a, b, c, d, 2};
    ASSERT_TRUE(m == (Mat2{a, b, c, d, 2}));
    ASSERT_FALSE(m == (Mat2{zero, b, c, d, 2}));
    ASSERT_FALSE(m == (Mat2{a, zero, c, d, 2}));
    ASSERT_FALSE(m == (Mat2{a, b, zero, d, 2}));
    ASSERT_FALSE(m == (Mat2{a, b, c, zero, 2}));
    ASSERT_FALSE(m == (Mat2{a, b, c, d, 14}));
    ASSERT_FALSE(m != (Mat2{a, b, c, d, 2}));
    ASSERT_EQ(m.str(), R"MAT({
 {1+2i+3√2+4i√2, 5+8i+11√2+14i√2},
 {6+9i+12√2+15i√2, 7+10i+13√2+16i√2}
} / 2**2)MAT");
    EXPECT_NEAR_COMPLEX22(m.approx(), (std::array<std::array<std::complex<double>, 2>, 2>{
        std::array<std::complex<double>, 2>{
            std::complex<double>{1.3106601717798214, 1.9142135623730951},
            std::complex<double>{5.139087296526012, 6.949747468305833}},
        std::array<std::complex<double>, 2>{
            std::complex<double>{5.742640687119286, 7.553300858899107},
            std::complex<double>{6.34619407771256, 8.15685424949238}},
    }), 1e-10);
}

TEST(mat2, add) {
    Mat2 m1{
        .a={.r=1, .i=5, .r2=9, .i2=13},
        .b={.r=2, .i=6, .r2=10, .i2=14},
        .c={.r=3, .i=7, .r2=11, .i2=15},
        .d={.r=4, .i=8, .r2=12, .i2=16},
        .denom_power=2,
    };
    Mat2 m2{
        .a={.r=10, .i=50, .r2=90, .i2=1300},
        .b={.r=20, .i=60, .r2=1000, .i2=1400},
        .c={.r=30, .i=70, .r2=1100, .i2=1500},
        .d={.r=40, .i=80, .r2=1200, .i2=1600},
        .denom_power=2,
    };
    Mat2 m3{
        .a={.r=11, .i=55, .r2=99, .i2=1313},
        .b={.r=22, .i=66, .r2=1010, .i2=1414},
        .c={.r=33, .i=77, .r2=1111, .i2=1515},
        .d={.r=44, .i=88, .r2=1212, .i2=1616},
        .denom_power=2,
    };
    ASSERT_EQ(m1 + m2, m3);
}

TEST(mat2, canonicalized) {
    ASSERT_EQ((Mat2{
        .a={.r=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized(), (Mat2{
        .a={.r=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized());

    ASSERT_EQ((Mat2{
        .a={.r=2},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized(), (Mat2{
        .a={.r=1},
        .b={},
        .c={},
        .d={},
        .denom_power=1,
    }).canonicalized());

    ASSERT_EQ((Mat2{
        .a={.r=4},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized(), (Mat2{
        .a={.r=1},
        .b={},
        .c={},
        .d={},
        .denom_power=0,
    }).canonicalized());


    ASSERT_EQ((Mat2{
        .a={.r=8},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized(), (Mat2{
        .a={.r=2},
        .b={},
        .c={},
        .d={},
        .denom_power=0,
    }).canonicalized());

    ASSERT_EQ((Mat2{
        .a={.r=0, .i=0, .r2=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized(), (Mat2{
        .a={.r=0, .i=0, .r2=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    }).canonicalized());
}

TEST(mat2, add_power_difference) {
    Mat2 m1{
        .a={.r=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    };
    Mat2 m2{
        .a={.r=2},
        .b={},
        .c={},
        .d={},
        .denom_power=8,
    };
    Mat2 m3{
        .a={.r=2 + (1 << 6)},
        .denom_power=8,
    };
    ASSERT_EQ(m1 + m2, m3.canonicalized());
}

TEST(mat2, add_power_difference_combo) {
    Mat2 m1{
        .a={.r=0, .i=0, .r2=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    };
    Mat2 m2{
        .a={.r=0, .i=0, .r2=1},
        .b={},
        .c={},
        .d={},
        .denom_power=2,
    };
    Mat2 m3{
        .a={.r=0, .i=0, .r2=1},
        .b={},
        .c={},
        .d={},
        .denom_power=1,
    };
    ASSERT_EQ(m1 + m2, m3);
}

TEST(mat2, mul) {
    Mat2 m1{
        .a=Val{2},
        .b=Val{3},
        .c=Val{5},
        .d=Val{7},
        .denom_power=1,
    };
    Mat2 m2{
        .a=Val{11},
        .b=Val{13},
        .c=Val{17},
        .d=Val{19},
        .denom_power=2,
    };
    Mat2 m3{
        .a=Val{73},
        .b=Val{83},
        .c=Val{174},
        .d=Val{198},
        .denom_power=3,
    };
    ASSERT_EQ(m1 * m2, m3);
}

TEST(mat2, canonicalize_approx) {
    Mat2 m1{
        .a=Val{2},
        .b=Val{0},
        .c=Val{0},
        .d=Val{0},
        .denom_power=1,
    };
    EXPECT_NEAR_COMPLEX22(m1.approx(), m1.canonicalized().approx(), 1e-8);
}

TEST(mat2, mul_canonicalize) {
    Mat2 m1{
        .a=Val{2},
        .b=Val{0},
        .c=Val{0},
        .d=Val{0},
        .denom_power=5,
    };
    Mat2 m2{
        .a=Val{4},
        .b=Val{0},
        .c=Val{0},
        .d=Val{0},
        .denom_power=6,
    };
    Mat2 m3{
        .a=Val{1},
        .b=Val{0},
        .c=Val{0},
        .d=Val{0},
        .denom_power=11 - 3,
    };
    ASSERT_EQ(m1 * m2, m3);
}

TEST(mat2, approx_abs_determinant) {
    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{0},
        .c=Val{0},
        .d=Val{0},
        .denom_power=0,
    }).approx_abs_determinant(), 0, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{0},
        .c=Val{0},
        .d=Val{1},
        .denom_power=0,
    }).approx_abs_determinant(), 1, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{0},
        .c=Val{0},
        .d=Val{2},
        .denom_power=0,
    }).approx_abs_determinant(), 2, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{0},
        .c=Val{0},
        .d=Val{1},
        .denom_power=1,
    }).approx_abs_determinant(), 0.25, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{1},
        .c=Val{1},
        .d=Val{1},
        .denom_power=1,
    }).approx_abs_determinant(), 0, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{1},
        .c=Val{1},
        .d=Val{-1},
        .denom_power=0,
    }).approx_abs_determinant(), 2, 1e-8);

    ASSERT_NEAR((Mat2{
        .a=Val{1},
        .b=Val{1},
        .c=Val{1},
        .d=Val{-1},
        .denom_power=1,
    }).approx_abs_determinant(), 0.5, 1e-8);
}

TEST(mat2, hadamard_determinant_product) {
    Val l{1};
    Mat2 H = Mat2{l, l, l, -l};
    ASSERT_NEAR(H.approx_abs_determinant(), 2, 1e-8);
    ASSERT_EQ(H * H, (Mat2{Val{2}, {}, {}, Val{2}}));
    ASSERT_NEAR((H * H).approx_abs_determinant(), 4, 1e-8);
}
