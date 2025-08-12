#ifndef _DECOMP_KET_H
#define _DECOMP_KET_H

#include <array>
#include <string_view>

#include "val.h"
#include "val_r2.h"

namespace decomp {

struct Ket {
    Val amp0;
    Val amp1;
    uint32_t divisions_by_two = 0;

    Ket canonicalized() const;
    void inplace_canonicalize();
    double approx_norm2() const;
    double infidelity_versus_rotation_power(size_t power) const;
    double infidelity(const Ket &other) const;
    Ket after_h_xy() const;
    Ket after_h_yz() const;
    Ket after_h_then_t() const;
    Ket after_h_then_t_dag() const;
    Ket after_h() const;
    Ket after_s() const;
    Ket after_z() const;
    Ket after_x() const;
    Ket after_t() const;
    Ket after_t_dag() const;
    static Ket ket_0();
    static Ket ket_1();
    static Ket ket_p();
    static Ket ket_m();
    static Ket ket_i();
    static Ket ket_j();
    Ket after_gate(std::string_view gate) const;

    std::array<ValR2, 3> bloch_vec() const;

    bool operator==(const Ket &other) const;
    Ket operator*(const Val &other) const;
    Ket over2() const;
    Ket operator-() const;
};

std::ostream &operator<<(std::ostream &out, const Ket &s);

}

#endif
