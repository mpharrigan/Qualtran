#include "state.h"

using namespace decomp;

static constexpr double SQRT_HALF = 0.70710678118654752440084436210484903928483593;

std::vector<std::string> decomp::reconstruct_gate_sequence(uint64_t address1, uint8_t clifford1, uint64_t address2, uint8_t clifford2) {
    if (address1 == 0 || address2 == 0) {
        throw std::invalid_argument("no address");
    }
    uint64_t c = 1ULL << (63ULL - std::countl_zero(address1));
    c >>= 1;
    std::vector<std::string> result;
    if (address1 & c) {
        result.push_back("H_YZ");
    }
    for (c >>= 1; c; c >>= 1) {
        if (address1 & c) {
            result.push_back("H");
            result.push_back("T_DAG");
        } else {
            result.push_back("H");
            result.push_back("T");
        }
    }
    if (clifford1 & 1) {
        result.push_back("H_YZ");
    }
    if (clifford1 & 2) {
        result.push_back("H_XY");
    }
    if (clifford1 & 4) {
        result.push_back("H_YZ");
    }
    if (clifford1 & 8) {
        result.push_back("Z");
    }
    if (clifford1 & 16) {
        result.push_back("X");
    }
    if (clifford2 & 16) {
        result.push_back("X");
    }
    if (clifford2 & 8) {
        result.push_back("Z");
    }
    if (clifford2 & 4) {
        result.push_back("H_YZ");
    }
    if (clifford2 & 2) {
        result.push_back("H_XY");
    }
    if (clifford2 & 1) {
        result.push_back("H_YZ");
    }
    result.push_back("H");
    size_t n = 63ULL - std::countl_zero(address2);
    for (size_t k = 0; k < n; k++) {
        if (address2 & (1 << k)) {
            result.push_back("H");
            result.push_back("T");
        } else {
            result.push_back("H");
            result.push_back("T_DAG");
        }
    }
    result.push_back("H");
    return result;
}

State State::approx(const Ket &other) {
    auto a0 = other.amp0.approx();
    auto a1 = other.amp1.approx();
    for (size_t k = 0; k < other.divisions_by_two; k++) {
        a0 *= 0.5;
        a1 *= 0.5;
    }
    return State{a0, a1, 0};
}

double State::norm2() const {
    return (amp0 * std::conj(amp0) + amp1 * std::conj(amp1)).real();
}

double State::infidelity(const State &other) const {
    auto v = amp0 * std::conj(other.amp0) + amp1 * std::conj(other.amp1);
    return 1 - (v * std::conj(v)).real();
}

State State::after_h_xy() const {
    return after_x().after_s();
}

State State::after_h_yz() const {
    auto a = amp0 + amp1*std::complex<double>{0, -1};
    auto b = amp0*std::complex<double>{0, 1} - amp1;
    return {a * SQRT_HALF, b * SQRT_HALF, address};
}

State State::after_h_then_t() const {
    std::complex<double> a = amp0 + amp1;
    std::complex<double> b = amp0 - amp1;
    a *= SQRT_HALF;
    b *= std::complex<double>{0.5, 0.5};
    return {a, b, address << 1};
}

State State::after_h_then_t_dag() const {
    std::complex<double> a = amp0 + amp1;
    std::complex<double> b = amp0 - amp1;
    a *= SQRT_HALF;
    b *= std::complex<double>{0.5, -0.5};
    return {a, b, (address << 1) | 1};
}

State State::after_h() const {
    std::complex<double> a = amp0 + amp1;
    std::complex<double> b = amp0 - amp1;
    a *= SQRT_HALF;
    b *= SQRT_HALF;
    return {a, b, address};
}

State State::after_s() const {
    std::complex<double> a = amp0;
    std::complex<double> b = amp1;
    b *= std::complex<double>{0, 1};
    return {a, b, address};
}

State State::after_z() const {
    std::complex<double> a = amp0;
    std::complex<double> b = amp1;
    return {a, -b, address};
}

State State::after_x() const {
    std::complex<double> a = amp0;
    std::complex<double> b = amp1;
    return {b, a, address};
}

std::vector<State> State::generalized_states() const {
    std::vector<State> result;
    result.push_back(*this);
    size_t n = result.size();
    for (size_t k = 0; k < n; k++) {
        result.push_back(result[k].after_h());
    }
    n = result.size();
    for (size_t k = 0; k < n; k++) {
        result.push_back(result[k].after_s());
    }
    n = result.size();
    for (size_t k = 0; k < n; k++) {
        result.push_back(result[k].after_h());
    }
    n = result.size();
    for (size_t k = 0; k < n; k++) {
        result.push_back(result[k].after_x());
    }
    n = result.size();
    for (size_t k = 0; k < n; k++) {
        result.push_back(result[k].after_z());
    }
    return result;
}

double State::generalized_infidelity(const State &other) const {
    double inf = infidelity(other);
    for (const auto &e : generalized_states()) {
        inf = std::min(inf, e.infidelity(other));
    }
    return inf;
}

State State::seed0() {
    return {
        .amp0=std::complex<double>{1, 0},
        .amp1=std::complex<double>{0, 0},
        .address=0b10,
    };
}

State State::seed1() {
    return {
        .amp0=std::complex<double>{SQRT_HALF, 0},
        .amp1=std::complex<double>{0, SQRT_HALF},
        .address=0b11,
    };
}

State State::extended_by_address(uint64_t extension_address) const {
    uint64_t c = 1ULL << (63ULL - std::countl_zero(extension_address));
    State v = *this;
    for (c >>= 1; c; c >>= 1) {
        if (extension_address & c) {
            v = v.after_h_then_t_dag();
        } else {
            v = v.after_h_then_t();
        }
    }
    return v;
}
State State::from_seed_address(uint64_t address) {
    uint64_t c = 1ULL << (63ULL - std::countl_zero(address));
    c >>= 1;
    State v = address & c ? seed1() : seed0();
    for (c >>= 1; c; c >>= 1) {
        if (address & c) {
            v = v.after_h_then_t_dag();
        } else {
            v = v.after_h_then_t();
        }
    }
    return v;
}
State State::reconstruct(uint64_t address1, uint8_t clifford1, uint64_t address2, uint8_t clifford2) {
    uint64_t c = 1ULL << (63ULL - std::countl_zero(address1));
    c >>= 1;
    State v = address1 & c ? seed1() : seed0();
    for (c >>= 1; c; c >>= 1) {
        if (address1 & c) {
            v = v.after_h_then_t_dag();
        } else {
            v = v.after_h_then_t();
        }
    }
    if (clifford1 & 1) {
        v = v.after_h_yz();
    }
    if (clifford1 & 2) {
        v = v.after_h_xy();
    }
    if (clifford1 & 4) {
        v = v.after_h_yz();
    }
    if (clifford1 & 8) {
        v = v.after_z();
    }
    if (clifford1 & 16) {
        v = v.after_x();
    }
    if (clifford2 & 16) {
        v = v.after_x();
    }
    if (clifford2 & 8) {
        v = v.after_z();
    }
    if (clifford2 & 4) {
        v = v.after_h_yz();
    }
    if (clifford2 & 2) {
        v = v.after_h_xy();
    }
    if (clifford2 & 1) {
        v = v.after_h_yz();
    }
    v = v.after_h();
    size_t n = 63ULL - std::countl_zero(address2);
    for (size_t k = 0; k < n; k++) {
        if (address2 & (1 << k)) {
            v = v.after_h_then_t();
        } else {
            v = v.after_h_then_t_dag();
        }
    }
    v = v.after_h();
    return v;
}

std::array<double, 3> State::bloch_vec() const {
    double r0 = amp0.real();
    double i0 = amp0.imag();
    double r1 = amp1.real();
    double i1 = amp1.imag();

    double dr = r0 - r1;
    double di = i0 - i1;
    double px = (dr*dr + di*di) * 0.5;

    double dri = r0 - i1;
    double dir = r1 + i0;
    double py = (dri*dri + dir*dir) * 0.5;

    double pz = r1*r1 + i1*i1;

    double x = 1 - 2*px;
    double y = 1 - 2*py;
    double z = 1 - 2*pz;
    return {x, y, z};
}

std::string State::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &decomp::operator<<(std::ostream &out, const State &s) {
    out << "{" << s.amp0 << ", " << s.amp1 << "} addr=0b";
    for (size_t k = 63; k--;) {
        out << "01"[(s.address & (1ULL << k)) != 0];
    }
    return out;
}
