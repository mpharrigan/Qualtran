#ifndef _DECOMP_MAT2_H
#define _DECOMP_MAT2_H

#include <array>
#include <cstdint>
#include <ostream>
#include <complex>

#include "val.h"

namespace decomp {

struct Mat2 {
    decomp::Val a, b, c, d;
    uint64_t denom_power = 0;

    Mat2 canonicalized() const;
    void inplace_canonicalize();
    Mat2 operator+(const Mat2 &other) const;
    Mat2 operator*(const Mat2 &other) const;
    Mat2 operator*(const Val &other) const;
    bool operator==(const Mat2 &other) const;
    Mat2 operator-() const;
    bool operator!=(const Mat2 &other) const;
    double approx_abs_determinant() const;
    std::array<std::array<std::complex<double>, 2>, 2> approx() const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const Mat2 &r);

}

#endif
