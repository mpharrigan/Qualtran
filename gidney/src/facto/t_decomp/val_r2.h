#ifndef _DECOMP_VAL_R2_H
#define _DECOMP_VAL_R2_H

#include <cstdint>
#include <ostream>

namespace decomp {

struct ValR2 {
    int64_t r = 0;
    int64_t r2 = 0;
    uint64_t divisions_by_2 = 0;

    ValR2 canonicalized() const;
    ValR2 operator*(const ValR2 &other) const;
    ValR2 operator+(const ValR2 &other) const;
    ValR2 operator-() const;
    ValR2 operator-(const ValR2 &other) const;
    double approx() const;
    bool operator==(const ValR2 &other) const;
};

std::ostream &operator<<(std::ostream &out, const ValR2 &v);

}

#endif
