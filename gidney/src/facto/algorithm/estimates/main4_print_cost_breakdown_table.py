from __future__ import annotations

import argparse
import collections
import math
from typing import Any

from facto.algorithm.estimates._cost_config import total_costs
from facto.algorithm.prep import ExecutionConfig, ProblemConfig, table_str
from scatter_script import CostKey


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    s = 6
    conf = ExecutionConfig.vacuous_config_for_problem(
        problem_conf=ProblemConfig.from_ini_content(
            f"""
    ; RSA2048 challenge number
    ; 2048 bits (617 digits)
    ; p=?
    ; q=?
    ; source: https://en.wikipedia.org/wiki/RSA_numbers#RSA-2048

    modulus = 25195908475657893494027183240048398571429282126204032027777137836043662020707595556264018525880784406918290641249515082189298559149176184502808489120072844992687392807287776735971418347270261896375014971824691165077613379859095700097330459748808428401797429100642458691817195118746121515172654632282216869987549182422433637259085141865462043576798423387184774447920739934236584823824281198163815010674810451660377306056201619676256133844143603833904414952634432190114657544454178424020924616515723350778707749817125772467962926386356373289912154831438167899885040445364023527381951378636564391212010397122822120720357
    generator = 3
    num_input_qubits = {math.ceil(2048 * (1 / 2 + 1 / s))}
    num_shots = {s + 1}
    parallelism = 1

    window1 = 6
    window3a = 3
    window3b = 3
    window4 = 4
    min_wraparound_gap = 32
    len_accumulator = 32
    mask_bits = 17
            """
        )
    )

    def clamp_cost_key(k: CostKey) -> CostKey:
        if k.name == "add" and k["n"] <= 35:
            return CostKey("add")
        if k.name == "qrom" and k["N"] <= 64:
            return CostKey("qrom")
        if k.name == "phase_lookup" and k["N"] <= 64:
            return CostKey("phase_lookup")
        return k

    table = {}
    costs: dict[str, collections.Counter[CostKey]] = total_costs(conf)
    costs: dict[str, dict[CostKey, int]] = {
        k: {clamp_cost_key(k2): v2 for k2, v2 in v.items()}
        for k, v in costs.items()
    }
    for k, v in costs.items():
        millis = 0
        toffolis = 0
        for k2, v2 in v.items():
            if k2.name == "add":
                n = k2.get("n", 35)
                millis += 2 * v2 * n / 35
                toffolis += v2 * n
            elif k2.name == "qrom":
                N = k2.get("N", 64)
                millis += 2 * v2 * (N / 2 ** 6)
                toffolis += v2 * max(0, N - N.bit_length() - 1)
            elif k2.name == "phase_lookup":
                N = k2.get("N", 64)
                millis += 1 * v2 * (N / 2 ** 6)
                N1 = N.bit_length() // 2
                N2 = N.bit_length() - N1
                toffolis += v2 * (max(0, 2 ** (N1 // 2 - N1 - 1)) + max(0, 2 ** (N2 // 2 - N2 - 1)))
            elif k2.name == "alloc_gradient":
                millis += 1
            elif k2.name == "iter":
                continue
            else:
                raise NotImplementedError(f"{k2=}")
        millis = int(millis)
        v[CostKey("millis")] = millis
        v[CostKey("tofs")] = toffolis
    total: collections.Counter[CostKey] = collections.Counter()
    for k, v in costs.items():
        for k2, v2 in v.items():
            total[k2] += v2
    costs["---"] = total
    costs["Total (per shot)"] = total
    costs["Total (per factoring)"] = {k: v * conf.num_shots for k, v in total.items()}
    all_keys = sorted({k for v in costs.values() for k in v.keys()},
                      key=lambda e: (e == CostKey('millis'), e))
    for row, (k, v) in enumerate(costs.items()):
        table[row * 1j + -1] = k
    divisors = {}
    for col, key in enumerate(all_keys):
        key2 = ""
        if key == CostKey("add"):
            key = "Additions"
            key2 = "(Millions)"
            divisors[col] = 10**6
        if key == CostKey("phase_lookup"):
            key = "Phase Lookups"
            key2 = "(Millions)"
            divisors[col] = 10**6
        if key == CostKey("qrom"):
            key = "QROM Lookups"
            key2 = "(Millions)"
            divisors[col] = 10**6
        if key == CostKey("millis"):
            key = "Duration"
            key2 = "(Hours)"
            divisors[col] = 60 * 60 * 1000
        if key == CostKey("tofs"):
            key = "Toffolis"
            key2 = "(millions)"
            divisors[col] = 10**6
        table[-3j + col] = key
        table[-2j + col] = key2
        table[-1j + col] = "---"
    table[-1j - 1] = "---"
    table[-2j - 1] = "Subroutine"
    table[-3j - 1] = f"RSA n={conf.modulus.bit_length()}"
    for row, (k, v) in enumerate(costs.items()):
        for col, key in enumerate(all_keys):
            if k == "---":
                table[row * 1j + col] = "---"
            elif key in v:
                table[row * 1j + col] = v[key] / divisors.get(col, 1)

    def latex_format(e: Any) -> Any:
        if isinstance(e, float) and 0.1 < e < 200:
            return str(round(e * 10) / 10)
        if isinstance(e, int) and e < 1000:
            return str(e)
        if isinstance(e, (int, float)):
            return f"{e:0.1e}"
        return e

    print(table_str(table, latex=args.latex, formatter=latex_format if args.latex else None))

    hot_distance = 25
    cold_storage_coding_rate = 440  # * 2 // 3
    cold_physicals = conf.num_input_qubits * cold_storage_coding_rate
    factory_width: int = 4
    factory_count: int = 6
    activity_area_width: int = 3
    hot_storage_hallway_length: int = 6
    hot_storage_back_rows: int = 4
    hot_height: int = 3 * factory_count
    hot_width: int = (
        factory_width + activity_area_width + hot_storage_hallway_length + hot_storage_back_rows
    )
    hot_coding_rate = (hot_distance + 1) ** 2 * 2
    hot_physicals = hot_width * hot_height * hot_coding_rate
    print(f"{hot_distance=}")
    print(f"{cold_storage_coding_rate=}")
    print(f"num_input_qubits={conf.num_input_qubits}")
    print(f"{cold_physicals=}")
    print(f"{hot_physicals=}")
    print("space/M", (hot_physicals + cold_physicals) / 10 ** 6)



if __name__ == "__main__":
    main()
