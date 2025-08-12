#ifndef _DECOMP_VAL_H
#define _DECOMP_VAL_H

#include <cstdint>
#include <ostream>
#include <complex>

namespace decomp {

struct Val {
    int64_t r = 0;
    int64_t i = 0;
    int64_t r2 = 0;
    int64_t i2 = 0;

    uint64_t sqrt2_divisibility() const;
    Val real() const;
    Val imag() const;
    Val operator+(const Val &other) const;
    Val operator-(const Val &other) const;
    Val operator*(const Val &other) const;
    Val operator<<(uint8_t shift) const;
    Val operator>>(uint8_t shift) const;
    Val &operator<<=(uint8_t shift);
    Val operator-() const;
    Val &operator>>=(uint8_t shift);
    Val times_sqrt2() const;
    Val squared_norm() const;
    Val conjugated() const;
    double approx_norm() const;
    bool operator==(const Val &other) const;
    bool operator!=(const Val &other) const;
    std::complex<double> approx() const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const Val &r);

}

#endif
