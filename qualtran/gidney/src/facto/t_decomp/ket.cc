#include "state.h"

using namespace decomp;

static constexpr double SQRT_HALF = 0.70710678118654752440084436210484903928483593;

Ket Ket::canonicalized() const {
    auto copy = *this;
    copy.inplace_canonicalize();
    return copy;
}

void Ket::inplace_canonicalize() {
    while (divisions_by_two > 0 && amp0.sqrt2_divisibility() >= 2 && amp1.sqrt2_divisibility() >= 2) {
        divisions_by_two -= 1;
        amp0 >>= 1;
        amp1 >>= 1;
    }
}

double Ket::approx_norm2() const {
    return (amp0 * amp0.conjugated() + amp1 * amp1.conjugated()).approx().real() / pow(4.0, divisions_by_two);
}

double Ket::infidelity(const Ket &other) const {
    auto a = canonicalized();
    auto b = other.canonicalized();
    while (a.divisions_by_two < b.divisions_by_two) {
        a.divisions_by_two += 1;
        a.amp0 <<= 1;
        a.amp1 <<= 1;
    }
    while (b.divisions_by_two < a.divisions_by_two) {
        b.divisions_by_two += 1;
        b.amp0 <<= 1;
        b.amp1 <<= 1;
    }

    auto v = a.amp0 * b.amp0.conjugated() + a.amp1 * b.amp1.conjugated();

    return (Val{.r=1} - v * v.conjugated()).approx().real() / pow(4.0, a.divisions_by_two);
}

Ket Ket::after_h_xy() const {
    return after_x().after_s();
}

Ket Ket::after_h_yz() const {
    Val a = amp0 + amp1*Val{.r=0, .i=-1};
    Val b = amp0*Val{.r=0, .i=1} - amp1;
    a = a.times_sqrt2();
    b = b.times_sqrt2();
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_h_then_t() const {
    Val a = amp0 + amp1;
    Val b = amp0 - amp1;
    a = a * Val{.r=0, .i=0, .r2=1};
    b = b * Val{.r=1, .i=1};
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_t() const {
    Val a = amp0 * Val{.r=2};
    Val b = amp1 * Val{.r=0,.i=0,.r2=1,.i2=1};
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_t_dag() const {
    Val a = amp0 * Val{.r=2};
    Val b = amp1 * Val{.r=0,.i=0,.r2=1,.i2=-1};
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_h_then_t_dag() const {
    Val a = amp0 + amp1;
    Val b = amp0 - amp1;
    a = a * Val{.r=0,.i=0,.r2=1};
    b = b * Val{.r=1, .i=-1};
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_h() const {
    Val a = amp0 + amp1;
    Val b = amp0 - amp1;
    a = a.times_sqrt2();
    b = b.times_sqrt2();
    return Ket{.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}

Ket Ket::after_s() const {
    Val a = amp0;
    Val b = amp1;
    b = b * Val{.r=0, .i=1};
    return {.amp0=a, .amp1=b, .divisions_by_two=divisions_by_two};
}

Ket Ket::after_z() const {
    Val a = amp0;
    Val b = amp1;
    return {.amp0=a, .amp1=-b, .divisions_by_two=divisions_by_two};
}

Ket Ket::after_x() const {
    Val a = amp0;
    Val b = amp1;
    return {.amp0=b, .amp1=a, .divisions_by_two=divisions_by_two};
}

Ket Ket::ket_0() {
    return {
        .amp0=Val{.r=1},
        .amp1=Val{},
    };
}
Ket Ket::ket_1() {
    return {
        .amp0=Val{},
        .amp1=Val{.r=1},
    };
}
Ket Ket::ket_p() {
    return {
        .amp0=Val{.r=0, .i=0, .r2=1},
        .amp1=Val{.r=0, .i=0, .r2=1},
        .divisions_by_two=1,
    };
}
Ket Ket::ket_m() {
    return {
        .amp0=Val{.r=0, .i=0, .r2=1},
        .amp1=Val{.r=0, .i=0, .r2=-1},
        .divisions_by_two=1,
    };
}
Ket Ket::ket_i() {
    return {
        .amp0=Val{.r=0, .i=0, .r2=1, .i2=0},
        .amp1=Val{.r=0, .i=0, .r2=0, .i2=1},
        .divisions_by_two=1,
    };
}
Ket Ket::ket_j() {
    return {
        .amp0=Val{.r=0, .i=0, .r2=1, .i2=0},
        .amp1=Val{.r=0, .i=0, .r2=0, .i2=-1},
        .divisions_by_two=1,
    };
}

Ket Ket::after_gate(std::string_view gate) const {
    if (gate == "H_XY") {
        return after_h_xy();
    } else if (gate == "H") {
        return after_h();
    } else if (gate == "H_YZ") {
        return after_h_yz();
    } else if (gate == "T") {
        return after_t();
    } else if (gate == "T_DAG") {
        return after_t_dag();
    } else if (gate == "X") {
        return after_x();
    } else if (gate == "Z") {
        return after_z();
    } else {
        std::stringstream ss;
        ss << "gate=";
        ss << gate;
        throw std::invalid_argument(ss.str());
    }
}

std::ostream &decomp::operator<<(std::ostream &out, const Ket &s) {
    out << "(" << s.amp0 << "|0> + " << s.amp1 << "|1>) / 2**" << s.divisions_by_two;
    return out;
}

Ket Ket::operator*(const Val &other) const {
    return (Ket{.amp0=amp0 * other, .amp1=amp1 * other, .divisions_by_two=divisions_by_two}).canonicalized();
}
Ket Ket::over2() const {
    return Ket{.amp0=amp0, .amp1=amp1, .divisions_by_two=divisions_by_two + 1}.canonicalized();
}
bool Ket::operator==(const Ket &other) const {
    return amp0 == other.amp0
        && amp1 == other.amp1
        && divisions_by_two == other.divisions_by_two;
}

Ket Ket::operator-() const {
    return {.amp0=-amp0, .amp1=-amp1, .divisions_by_two=divisions_by_two};
}

double Ket::infidelity_versus_rotation_power(size_t power) const {
    double theta = 3.14159265359 * pow(2.0, -(double)power);
    if (power < 15) {
        auto a0 = amp0.approx();
        auto a1 = amp1.approx();
        auto b0 = SQRT_HALF;
        auto b1 = std::complex<double>{cos(theta), sin(theta)} * SQRT_HALF;
        auto v = a0 * std::conj(b0) + a1 * std::conj(b1);
        return 1.0 - (v * std::conj(v)).real();
    }

    auto vec = bloch_vec();
    double x = vec[0].approx();
    double dx = x > 0.9999 ? 0 : (x - 1);
    double dy = vec[1].approx() - theta;
    double dz = vec[2].approx();
    return dx*dx + dy*dy + dz*dz;
}

std::array<ValR2, 3> Ket::bloch_vec() const {
    ValR2 r0 = ValR2{.r=amp0.r, .r2=amp0.r2, .divisions_by_2=divisions_by_two}.canonicalized();
    ValR2 i0 = ValR2{.r=amp0.i, .r2=amp0.i2, .divisions_by_2=divisions_by_two}.canonicalized();
    ValR2 r1 = ValR2{.r=amp1.r, .r2=amp1.r2, .divisions_by_2=divisions_by_two}.canonicalized();
    ValR2 i1 = ValR2{.r=amp1.i, .r2=amp1.i2, .divisions_by_2=divisions_by_two}.canonicalized();

    ValR2 dr = r0 - r1;
    ValR2 di = i0 - i1;
    ValR2 px = dr*dr + di*di;

    ValR2 dri = r0 - i1;
    ValR2 dir = r1 + i0;
    ValR2 py = dri*dri + dir*dir;

    ValR2 pz = r1*r1 + i1*i1;

    ValR2 x = ValR2{.r=1} - px;
    ValR2 y = ValR2{.r=1} - py;
    ValR2 z = ValR2{.r=1} - ValR2{.r=2}*pz;
    return {x, y, z};
}
