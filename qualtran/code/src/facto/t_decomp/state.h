#ifndef _DECOMP_STATE_H
#define _DECOMP_STATE_H

#include <cstdint>
#include <ostream>
#include <complex>
#include <vector>
#include <string_view>

#include "ket.h"
#include "val.h"
#include "val_r2.h"

namespace decomp {

std::vector<std::string> reconstruct_gate_sequence(uint64_t address1, uint8_t clifford1, uint64_t address2, uint8_t clifford2);

struct State {
    std::complex<double> amp0;
    std::complex<double> amp1;
    uint64_t address = 0;

    static State approx(const Ket &other);
    double norm2() const;
    double infidelity(const State &other) const;
    State after_h_xy() const;
    State after_h_yz() const;
    State after_h_then_t() const;
    State after_h_then_t_dag() const;
    State after_h() const;
    State after_s() const;
    State after_z() const;
    State after_x() const;
    std::vector<State> generalized_states() const;
    double generalized_infidelity(const State &other) const;
    static State seed0();
    static State seed1();
    State extended_by_address(uint64_t extension_address) const;
    static State from_seed_address(uint64_t address);
    static State reconstruct(uint64_t address1, uint8_t clifford1, uint64_t address2, uint8_t clifford2);
    std::array<double, 3> bloch_vec() const;
    std::string str() const;
};

std::ostream &operator<<(std::ostream &out, const State &s);

}

#endif
