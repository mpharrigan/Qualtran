#include "val_r2.h"

using namespace decomp;

constexpr double SQRT_2 = 1.4142135623730950488016887242096980785696718753769480731766797379907;

ValR2 ValR2::canonicalized() const {
    ValR2 copy = *this;
    while ((copy.r & 1) == 0 && (copy.r2 & 1) == 0 && copy.divisions_by_2 > 0) {
        copy.divisions_by_2 -= 1;
        copy.r >>= 1;
        copy.r2 >>= 1;
    }
    return copy;
}

double ValR2::approx() const {
    double result = r + r2*SQRT_2;
    for (size_t k = 0; k < divisions_by_2; k++) {
        result *= 0.5;
    }
    return result;
}

ValR2 ValR2::operator*(const ValR2 &other) const {
    return ValR2{
        .r=r*other.r + r2*other.r2*2,
        .r2=r*other.r2 + r2*other.r,
        .divisions_by_2=divisions_by_2 + other.divisions_by_2,
    }.canonicalized();
}
ValR2 ValR2::operator+(const ValR2 &other) const {
    ValR2 a = *this;
    ValR2 b = other;
    while (b.divisions_by_2 < a.divisions_by_2) {
        b.divisions_by_2 += 1;
        b.r <<= 1;
        b.r2 <<= 1;
    }
    while (a.divisions_by_2 < b.divisions_by_2) {
        a.divisions_by_2 += 1;
        a.r <<= 1;
        a.r2 <<= 1;
    }
    return ValR2{.r=a.r + b.r, .r2=a.r2 + b.r2, .divisions_by_2=a.divisions_by_2}.canonicalized();
}
ValR2 ValR2::operator-() const {
    return ValR2{.r=-r, .r2=-r2, .divisions_by_2=divisions_by_2};
}
bool ValR2::operator==(const ValR2 &other) const {
    return r == other.r && r2 == other.r2 && divisions_by_2 == other.divisions_by_2;
}
ValR2 ValR2::operator-(const ValR2 &other) const {
    return *this + -other;
}

std::ostream &decomp::operator<<(std::ostream &out, const ValR2 &val) {
    bool has_term = false;
    if (val.r) {
        out << val.r;
        has_term = true;
    }
    if (val.r2) {
        if (val.r2 > 0 && has_term) {
            out << "+";
        }
        if (val.r2 == -1) {
            out << "-";
        } else if (val.r2 != 1) {
            out << val.r2;
        }
        out << "√2";
        has_term = true;
    }
    if (!has_term) {
        out << "0";
    }
    if (val.divisions_by_2 > 0) {
        out << " / 2**" << val.divisions_by_2;
    }
    return out;
}
