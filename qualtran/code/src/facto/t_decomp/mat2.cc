#include <sstream>
#include <iostream>

#include "mat2.h"

using namespace decomp;

bool Mat2::operator==(const Mat2 &other) const {
    return a == other.a
        && b == other.b
        && c == other.c
        && d == other.d
        && denom_power == other.denom_power;
}

bool Mat2::operator!=(const Mat2 &other) const {
    return !(*this == other);
}

std::string Mat2::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::array<std::array<std::complex<double>, 2>, 2> Mat2::approx() const {
    double p = (double)(1ULL << denom_power);
    return std::array<std::array<std::complex<double>, 2>, 2>{
        std::array<std::complex<double>, 2>{a.approx() / p, b.approx() / p},
        std::array<std::complex<double>, 2>{c.approx() / p, d.approx() / p},
    };
}

Mat2 Mat2::operator+(const Mat2 &other) const {
    if (denom_power < other.denom_power) {
        return other + *this;
    }
    uint64_t m = denom_power - other.denom_power;
    Mat2 result{
        .a=a + (other.a << m),
        .b=b + (other.b << m),
        .c=c + (other.c << m),
        .d=d + (other.d << m),
        .denom_power=denom_power,
    };
    result.inplace_canonicalize();
    return result;
}

double Mat2::approx_abs_determinant() const {
    return (a * d - b * c).approx_norm() / pow(4.0, denom_power);
}

Mat2 Mat2::operator*(const Val &other) const {
    Mat2 result{
        .a=a*other,
        .b=b*other,
        .c=c*other,
        .d=d*other,
        .denom_power=denom_power,
    };
    result.inplace_canonicalize();
    return result;
}

Mat2 Mat2::operator*(const Mat2 &other) const {
    Mat2 result{
        .a=a*other.a + b*other.c,
        .b=a*other.b + b*other.d,
        .c=c*other.a + d*other.c,
        .d=c*other.b + d*other.d,
        .denom_power=denom_power + other.denom_power,
    };
    result.inplace_canonicalize();
    return result;
}

Mat2 Mat2::canonicalized() const {
    Mat2 copy = *this;
    copy.inplace_canonicalize();
    return copy;
}

void Mat2::inplace_canonicalize() {
    uint64_t ab = std::min(a.sqrt2_divisibility(), b.sqrt2_divisibility());
    uint64_t cd = std::min(c.sqrt2_divisibility(), d.sqrt2_divisibility());
    uint64_t abcd = std::min(ab, cd);
//    if (abcd & 1) {
//        a.inplace_times_sqrt2();
//        b.inplace_times_sqrt2();
//        c.inplace_times_sqrt2();
//        d.inplace_times_sqrt2();
//        abcd += 1;
//    }
    abcd >>= 1;
    abcd = std::min(abcd, denom_power);
    a >>= abcd;
    b >>= abcd;
    c >>= abcd;
    d >>= abcd;
    denom_power -= abcd;
}
Mat2 Mat2::operator-() const {
    return Mat2{
        .a=-a,
        .b=-b,
        .c=-c,
        .d=-d,
        .denom_power=denom_power,
    };
}

std::ostream &decomp::operator<<(std::ostream &out, const Mat2 &val) {
    out << "{\n {" << val.a;
    out << ", " << val.b;
    out << "},\n {" << val.c;
    out << ", " << val.d;
    out << "}\n}";
    if (val.denom_power) {
        out << " / 2**" << val.denom_power;
    }
    return out;
}
