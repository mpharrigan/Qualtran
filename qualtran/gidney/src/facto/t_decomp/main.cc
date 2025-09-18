#include <iostream>
#include <chrono>
#include <cassert>
#include <random>
#include <unordered_map>

#include "state.h"

using namespace decomp;

constexpr double SQRT_HALF = 0.70710678118654752440084436210484903928483593;


struct AddressState {
    double x;
    double y;
    uint64_t address;
    uint8_t clifford;

    double infidelity_proxy(const AddressState &other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return dx*dx + dy*dy;
    }

    static AddressState from_state(State state) {
        auto xyz = state.bloch_vec();
        auto x = xyz[0];
        auto y = xyz[1];
        auto z = xyz[2];
        uint8_t clifford = 0;

        // Sort YZ via H_YZ gate.
        if (abs(z) > abs(y)) {
            std::swap(z, y);
            x *= -1;
            clifford |= 1 << 0;
        }

        // Sort XY via H_XY gate.
        if (abs(y) > abs(x)) {
            std::swap(x, y);
            z *= -1;
            clifford |= 1 << 1;
        }

        // Sort YZ via H_YZ gate.
        if (abs(z) > abs(y)) {
            std::swap(z, y);
            x *= -1;
            clifford |= 1 << 2;
        }

        // Force X positive via Z gate.
        if (x < 0) {
            x *= -1;
            y *= -1;
            clifford |= 1 << 3;
        }

        // Force Y positive via X gate.
        if (y < 0) {
            y *= -1;
            z *= -1;
            clifford |= 1 << 4;
        }

        // Now guaranteed: |z| <= y <= x

        double cx = atan2(y, x) / (3.14159265359 / 2.0);
        double cy = z * sqrt(3.0 / 4.0);
        return AddressState{.x=cx, .y=cy, .address=state.address, .clifford=clifford};
    }

    uint64_t key_address(uint32_t buckets) const {
        return ((uint64_t)floor(x * buckets) << 32) | (uint64_t)floor(y * buckets);
    }
};

inline uint64_t next_address(uint64_t start_address, uint64_t address, uint64_t collisions) {
    return start_address * 910913378854189 + address * 15621151774324842563ULL + 11381484653730381876ULL + collisions;
}

struct AddressMap {
    size_t buckets;
    std::unordered_map<uint64_t, AddressState> vals;
    size_t insertion_attempts = 0;
    size_t insertion_collisions = 0;
    size_t insertion_skips = 0;

    AddressMap(size_t buckets) : buckets(buckets) {
    }

    static AddressMap from_brute_force_sweep(size_t buckets, size_t levels, double insertion_atol) {
        AddressMap result(buckets);

        std::vector<State> states;
        states.reserve(1ULL << (levels + 2));
        states.push_back(State::seed0());
        states.push_back(State::seed1());
        size_t cur = 0;
        for (size_t level = 0; level < levels; level++) {
            size_t n = states.size();
            while (cur < n) {
                State state = states[cur];
                assert(State::from_seed_address(state.address).amp0 == state.amp0);
                states.push_back(state.after_h_then_t());
                states.push_back(state.after_h_then_t_dag());
                cur++;
            }
        }
        for (const auto &s : states) {
            result.ensure_included(s, insertion_atol);
        }

        return result;
    }

    template <typename CALLBACK>
    inline void lookup_address(uint64_t start_address, CALLBACK callback) {
        uint64_t address = start_address;
        uint64_t collisions = 0;
        while (true) {
            auto f = vals.find(address);
            if (f == vals.end()) {
                break;
            }
            const AddressState &out = f->second;
            callback(out);
            collisions += 1;
            address = next_address(start_address, address, collisions);
        }
    }

    template <typename CALLBACK>
    inline void lookup(AddressState key, CALLBACK callback) {
        uint64_t address = key.key_address(buckets);
        lookup_address(address - 0x100000000 - 1, callback);
        lookup_address(address - 0x100000000 - 0, callback);
        lookup_address(address - 0x100000000 + 1, callback);
        lookup_address(address - 0x000000000 - 1, callback);
        lookup_address(address - 0x000000000 - 0, callback);
        lookup_address(address - 0x000000000 + 1, callback);
        lookup_address(address + 0x100000000 - 1, callback);
        lookup_address(address + 0x100000000 - 0, callback);
        lookup_address(address + 0x100000000 + 1, callback);
    }

    void ensure_included(State state, double atol) {
        insertion_attempts += 1;
        AddressState addr_state = AddressState::from_state(state);
        uint64_t start_address = addr_state.key_address(buckets);
        uint64_t address = start_address;
        uint64_t collisions = 0;
        while (true) {
            auto f = vals.find(address);
            if (f == vals.end()) {
                break;
            }
            if (addr_state.infidelity_proxy(f->second) <= atol) {
                insertion_skips += 1;
                return;
            }
            collisions += 1;
            address = next_address(start_address, address, collisions);
        }
        insertion_collisions += collisions;
        vals.insert({address, addr_state});
    }
};

struct OutState {
    uint64_t address1 = 0;
    uint8_t clifford1 = 0;
    uint64_t address2 = 0;
    uint8_t clifford2 = 0;
    double score = 100;
};

int main(int argc, const char **argv) {
    if (argc != 4) {
        std::cerr << "Expected 3 command line arguments:\n";
        std::cerr << "    - log2_index_size: controls size of spatial lookup table.\n";
        std::cerr << "    - max_t_count: Max T gates to use in decompositions.\n";
        std::cerr << "    - num_phase_gradient_qubits: Which phase states to try to generate.\n";
        return 1;
    }


    long a1 = atol(argv[1]);
    long a2 = atol(argv[2]);
    long a3 = atol(argv[3]);
    if (a1 <= 0 || a1 >= 40) {
        std::cerr << "Invalid log2_index_size\n";
        return 1;
    }
    if (a2 <= 0 || a2 >= 100) {
        std::cerr << "Invalid max_t_count\n";
        return 1;
    }
    if (a3 <= 0 || a3 >= 100) {
        std::cerr << "Invalid num_phase_gradient_qubits\n";
        return 1;
    }
    size_t buckets = 1 << a1;
    size_t levels0 = a2 >> 1;
    size_t levels = a2 - levels0;
    size_t powers = a3;
    std::cerr << "buckets=" << buckets << "\n";
    std::cerr << "max_t_count=" << (levels0 + levels) << "\n";
    std::cerr << "num_phase_gradient_qubits=" << powers << "\n";

    std::cerr << "building lookup map...\n";
    AddressMap map = AddressMap::from_brute_force_sweep(buckets, levels0, 1e-15);

    std::cerr << "elided collision rate " << (map.insertion_skips / (double)map.insertion_attempts) << "\n";
    std::cerr << "storage collision rate " << (map.insertion_collisions / (double)map.insertion_attempts) << "\n";

    std::cout << "gradient_qubit_index,t_count,infidelity,gates\n";
    for (size_t power = 0; power < powers; power++) {
        double theta = 3.14159265359 * pow(2.0, -(double)power);
        State target_state{
            .amp0=SQRT_HALF,
            .amp1=std::complex<double>{cos(theta), sin(theta)} * SQRT_HALF,
            .address=1,
        };

        std::array<OutState, 64> bests;
        std::vector<State> states;
        states.reserve(1ULL << (levels + 1));
        states.push_back(target_state);
        size_t cur = 0;
        for (size_t level = 0; level < levels; level++) {
            size_t n = states.size();
            while (cur < n) {
                State state = states[cur];
                states.push_back(state.after_h_then_t());
                states.push_back(state.after_h_then_t_dag());
                cur++;
            }
        }
        for (const auto &s : states) {
            AddressState addr_state = AddressState::from_state(s);
            map.lookup(addr_state, [&](const AddressState &other) {
                uint64_t t_count = 126ULL - std::countl_zero(other.address) - std::countl_zero(addr_state.address);
                double d = addr_state.infidelity_proxy(other);
                if (d < bests[t_count].score) {
                    bests[t_count].score = d;
                    bests[t_count].address1 = other.address;
                    bests[t_count].clifford1 = other.clifford;
                    bests[t_count].address2 = addr_state.address;
                    bests[t_count].clifford2 = addr_state.clifford;
                }
            });
        }

        double best_seen = 0.1;
        for (size_t kp = 0; kp < bests.size() && best_seen > 0; kp++) {
            if (bests[kp].address1 == 0) {
                continue;
            }
            auto seq = reconstruct_gate_sequence(bests[kp].address1, bests[kp].clifford1, bests[kp].address2, bests[kp].clifford2);
            size_t t_count = 0;
            Ket ex = Ket::ket_0();
            for (const auto &g : seq) {
                ex = ex.after_gate(g);
                if (g == "T" || g == "T_DAG") {
                    t_count += 1;
                }
            }
            double actual_inf = ex.infidelity_versus_rotation_power(power);
            if (actual_inf >= best_seen) {
                continue;
            }
            best_seen = actual_inf;

            std::cout << power << ",";
            std::cout << t_count << ",";
            std::cout << sqrt(abs(actual_inf)) << ",";
            for (size_t k2 = 0; k2 < seq.size(); k2++) {
                if (k2 > 0) {
                    std::cout << "+";
                }
                std::cout << seq[k2];
            }
            std::cout << "\n";
        }
    }

    return 0;
}
