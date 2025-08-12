#include <sstream>

#include "val.h"

using namespace decomp;

bool Val::operator==(const Val &other) const {
    return r == other.r && i == other.i && r2 == other.r2 && i2 == other.i2;
}

bool Val::operator!=(const Val &other) const {
    return !(*this == other);
}

uint64_t Val::sqrt2_divisibility() const {
    uint64_t nr = (uint64_t)std::countr_zero((uint64_t)r);
    uint64_t ni = (uint64_t)std::countr_zero((uint64_t)i);
    uint64_t nr2 = (uint64_t)std::countr_zero((uint64_t)r2);
    uint64_t ni2 = (uint64_t)std::countr_zero((uint64_t)i2);
    uint64_t n = std::min(nr, ni);
    uint64_t n2 = std::min(nr2, ni2);
    return std::min(2*n, 2*n2 + 1);
}

std::string Val::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

Val Val::operator+(const Val &other) const {
    return Val{
        .r=r+other.r,
        .i=i+other.i,
        .r2=r2+other.r2,
        .i2=i2+other.i2,
    };
}

Val Val::conjugated() const {
    return Val{.r=r, .i=-i, .r2=r2, .i2=-i2};
}

Val Val::squared_norm() const {
    return *this * this->conjugated();
}

double Val::approx_norm() const {
    return sqrt(squared_norm().approx().real());
}

Val Val::operator-(const Val &other) const {
    return Val{
        .r=r-other.r,
        .i=i-other.i,
        .r2=r2-other.r2,
        .i2=i2-other.i2,
    };
}

Val Val::operator*(const Val &other) const {
    return Val{
        .r=r*other.r - i*other.i + 2*r2*other.r2 - 2*i2*other.i2,
        .i=r*other.i + i*other.r + 2*r2*other.i2 + 2*i2*other.r2,
        .r2=r*other.r2 + r2*other.r - i2*other.i - i*other.i2,
        .i2=r2*other.i + i*other.r2 + i2*other.r + r*other.i2,
    };
}

Val Val::operator<<(uint8_t shift) const {
    return Val{
        .r=r << shift,
        .i=i << shift,
        .r2=r2 << shift,
        .i2=i2 << shift,
    };
}

Val Val::operator>>(uint8_t shift) const {
    return Val{
        .r=r >> shift,
        .i=i >> shift,
        .r2=r2 >> shift,
        .i2=i2 >> shift,
    };
}

Val &Val::operator>>=(uint8_t shift) {
    r >>= shift;
    i >>= shift;
    r2 >>= shift;
    i2 >>= shift;
    return *this;
}

Val Val::operator-() const {
    return {
        .r=-r,
        .i=-i,
        .r2=-r2,
        .i2=-i2,
    };
}

Val &Val::operator<<=(uint8_t shift) {
    r <<= shift;
    i <<= shift;
    r2 <<= shift;
    i2 <<= shift;
    return *this;
}

Val Val::times_sqrt2() const {
    return Val{
        .r=2 * r2,
        .i=2 * i2,
        .r2=r,
        .i2=i,
    };
}

Val Val::real() const {
    return Val{.r=r, .i=0, .r2=r2, .i2=0};
}

Val Val::imag() const {
    return Val{.r=0, .i=i, .r2=0, .i2=i2};
}

std::complex<double> Val::approx() const {
    double s = sqrt(2.0);
    return std::complex<double>{
        (double)r + s*(double)r2,
        (double)i + s*(double)i2,
    };
}

std::ostream &decomp::operator<<(std::ostream &out, const Val &val) {
    bool has_term = false;
    if (val.r) {
        out << val.r;
        has_term = true;
    }
    if (val.i) {
        if (val.i > 0 && has_term) {
            out << "+";
        }
        if (val.i == -1) {
            out << "-";
        } else if (val.i != 1) {
            out << val.i;
        }
        out << "i";
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
    if (val.i2) {
        if (val.i2 > 0 && has_term) {
            out << "+";
        }
        if (val.i2 == -1) {
            out << "-";
        } else if (val.i2 != 1) {
            out << val.i2;
        }
        out << "i√2";
        has_term = true;
    }
    if (!has_term) {
        out << "0";
    }
    return out;
}
