from __future__ import annotations

import collections
import dataclasses
import math

from facto.algorithm.prep import ExecutionConfig, simplified_costs, prime_count_and_capacity_at_bit_length
from scatter_script import CostKey


def total_costs(conf: ExecutionConfig) -> dict[str, collections.Counter[CostKey]]:
    return {
        subroutine: simplified_costs(v) for subroutine, v in conf.estimated_subroutine_costs.items()
    }


@dataclasses.dataclass(frozen=True)
class CostConfig:
    s: int
    conf: ExecutionConfig
    toffolis: int
    keep_rate: float

    @staticmethod
    def from_params(
        *, n: int, l: int, s: int, w1: int, w3: int, w4: int, len_acc: int
    ) -> CostConfig | None:
        x_size = math.ceil(n * (1 / 2 + 1 / (2 * s)))
        y_size = math.ceil(n / (2 * s))
        m = x_size + y_size
        num_mults = math.ceil(m / w1)
        cap = num_mults * n
        num_periods = math.ceil(cap / l)
        count, cap = prime_count_and_capacity_at_bit_length(l)
        if num_periods > count:
            return None
        if cap < n * num_mults:
            return None
        conf = ExecutionConfig.vacuous_config(
            num_shots=s + 1,
            modulus_bitlength=n,
            num_input_qubits=m,
            num_periods=num_periods,
            period_bitlength=l,
            w1=w1,
            w3a=w3,
            w3b=w3,
            w4=w4,
            len_acc=len_acc,
        )
        if conf.probability_of_deviation_failure > 0.9:
            return None

        costs = total_costs(conf)
        toffolis = 0
        for k1, v1 in costs.items():
            k2: CostKey
            for k2, v2 in v1.items():
                if k2.name == "add":
                    n = k2.get("n", 35)
                    toffolis += v2 * n
                elif k2.name == "qrom":
                    N = k2.get("N", 64)
                    toffolis += v2 * max(0, N - N.bit_length() - 1)
                elif k2.name == "phase_lookup":
                    N = k2.get("N", 64)
                    N1 = N.bit_length() // 2
                    N2 = N.bit_length() - N1
                    toffolis += v2 * (
                        max(0, 2 ** (N1 // 2 - N1 - 1)) + max(0, 2 ** (N2 // 2 - N2 - 1))
                    )
                elif k2.name == "alloc_gradient":
                    continue
                else:
                    raise NotImplementedError(f"{k2=}")
        keep_rate = 1 - conf.probability_of_deviation_failure
        toffolis = math.ceil(toffolis * conf.num_shots / keep_rate)
        return CostConfig(s=s, conf=conf, toffolis=toffolis, keep_rate=keep_rate)

    @staticmethod
    def iter_configurations(tup) -> str:
        result = []
        seen = set()
        n, s, len_acc, l = tup
        for w1 in range(2, 9)[::-1]:
            for w3 in range(2, 6)[::-1]:
                for w4 in range(2, 9)[::-1]:
                    tt: CostConfig = CostConfig.from_params(n=n, s=s, l=l, w1=w1, w3=w3, w4=w4, len_acc=len_acc)
                    if tt is not None:
                        key = (tt.toffolis, tt.conf.estimated_logical_qubits)
                        if key in seen:
                            continue
                        seen.add(key)
                        result.append(
                            f"""{n},{s},{l},{w1},{w3},{w4},{len_acc},{tt.toffolis},{tt.conf.estimated_logical_qubits}"""
                        )
        return "\n".join(result)
