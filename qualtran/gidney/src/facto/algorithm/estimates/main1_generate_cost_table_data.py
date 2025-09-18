from __future__ import annotations

import argparse
import itertools
import multiprocessing
import pathlib
import sys
from typing import Any

from facto.algorithm.estimates._cost_config import CostConfig


def generate_cost_table_data(
        *,
        out_csv_path: str | pathlib.Path,
        min_n: int,
        max_n: int,
        power_of_2_multipliers: list[float],
):
    pool = multiprocessing.Pool()
    ns = sorted(
        round(m * 2**k)
        for k in range(min_n.bit_length() - 3, max_n.bit_length() + 3)
        for m in power_of_2_multipliers
        if min_n <= round(m * 2**k) <= max_n
    )
    print(f"Generating grid scan for {ns=}...", file=sys.stderr)
    tups = itertools.product(
        ns,
        range(2, 15),  # s
        range(24, 60),  # len_acc
        range(18, 26),  # l
    )
    f: Any
    with open(out_csv_path, 'w') as f:
        print('n,s,l,w1,w3,w4,len_acc,tofs,qubits', file=f)
        for lines in pool.imap(CostConfig.iter_configurations, tups):
            print(lines, file=f)
    print(f"wrote file://{pathlib.Path(out_csv_path).absolute()}", file=sys.stderr)


def read_buckets(csv_path: str | pathlib.Path) -> dict[int, dict[int, tuple[int, ...]]]:
    buckets: dict[int, dict[int, tuple[int, ...]]] = {}
    f: Any
    with open(csv_path) as f:
        f.readline()
        for line in f.readlines():
            if line.strip():
                n, s, l, w1, w3, w4, len_acc, tofs, qubits = line.split(',')
                tofs = int(tofs)
                n = int(n)
                qubits = int(qubits)
                if n not in buckets:
                    buckets[n] = {}
                if qubits not in buckets[n] or tofs < buckets[n][qubits][0]:
                    buckets[n][qubits] = (int(tofs), int(n), int(s), int(l), int(w1), int(w3), int(w4), int(len_acc))
    return buckets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, type=str)
    parser.add_argument('--min_n', required=True, type=int)
    parser.add_argument('--max_n', required=True, type=int)
    parser.add_argument('--power_of_2_multipliers', required=True, type=float, nargs='+')
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    csv_path.parent.mkdir(exist_ok=True, parents=True)
    generate_cost_table_data(
        out_csv_path=csv_path,
        min_n=args.min_n,
        max_n=args.max_n,
        power_of_2_multipliers=args.power_of_2_multipliers,
    )


if __name__ == "__main__":
    main()
