import collections
import math
from typing import Iterator, Iterable, Sequence, Callable

import sympy


def iter_primes() -> Iterator[int]:
    sieve = sympy.Sieve()
    n = 0
    seen = set()
    while True:
        n += 1000
        sieve.extend_to_no(n)
        for i in sieve._list:
            if i not in seen:
                seen.add(i)
                yield i


def iter_set_bits_of(x: int) -> Iterator[int]:
    assert x >= 0
    while x:
        yield (x ^ (x - 1)).bit_length() - 1
        x &= x - 1


def ceil_lg2(x: int) -> int:
    return (x - 1).bit_length()


def find_multiplicative_generator_modulo_prime_number(prime_number: int) -> int:
    # Adapted from https://gitlab.inria.fr/capsule/quantum-factoring-less-qubits/-/blob/main/modular_multi_product.py
    divisors = sympy.divisors(prime_number - 1)[:-1]
    for i in range(3, prime_number):
        if all(pow(i, d, prime_number) != 1 for d in divisors):
            return i
    raise ValueError(f"Failed to find a generator {prime_number=}.")


_HARDCODED_PRIME_STATS = {
    0: (0, 0),
    1: (0, 0),
    2: (2, 2),
    3: (2, 5),
    4: (2, 7),
    5: (5, 22),
    6: (7, 39),
    7: (13, 84),
    8: (23, 173),
    9: (43, 367),
    10: (75, 716),
    11: (137, 1444),
    12: (255, 2945),
    13: (464, 5823),
    14: (872, 11817),
    15: (1612, 23457),
    16: (3030, 47117),
    17: (5709, 94496),
    18: (10749, 188670),
    19: (20390, 378296),
    20: (38635, 755437),
    21: (73586, 1512435),
    22: (140336, 3024742),
    23: (268216, 6049260),
    24: (513708, 12099777),
}


def prime_count_and_capacity_at_bit_length(bit_length: int) -> tuple[int, int]:
    if bit_length in _HARDCODED_PRIME_STATS:
        # Precomputed exact result.
        return _HARDCODED_PRIME_STATS[bit_length]

    # Estimate using the prime number theorem.
    n1 = 1 << bit_length
    n2 = 1 << (bit_length - 1)
    count = math.ceil(n1 / math.log(n1) - n2 / math.log(n2))
    cap = math.ceil(n1 / math.log(4))
    return count, cap


def sig_fig_str(x: float, *, suffixes: Sequence[str] = ("", "K", "M", "B", "T")):
    suffix_index = 0
    while x >= 10_000:
        x //= 1000
        suffix_index += 1
    if x >= 1000:
        suffix_index += 1
        x //= 100
        result = f"{x / 10}"
    else:
        result = f"{x}"
    if suffix_index >= len(suffixes):
        result += f"E{suffix_index*3}"
    else:
        result += suffixes[suffix_index]
    return result.rjust(4, " ")


def choose(n: int, k: int) -> int:
    assert 0 <= k <= n
    if 2 * k >= n:
        k = n - k
    result = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    for i in range(2, k + 1):
        result //= i
    return result


def gcd_ex(a: int, b: int) -> tuple[int, int, int]:
    """Returns (x, y, g) such that x*a + y*b == g math.gcd(x, y)."""
    p1 = 1
    q1 = 0
    h1 = a
    p2 = 0
    q2 = 1
    h2 = b

    while h2 != 0:
        r = h1 // h2
        p3 = p1 - r * p2
        q3 = q1 - r * q2
        h3 = h1 - r * h2
        p1 = p2
        q1 = q2
        h1 = h2
        p2 = p3
        q2 = q3
        h2 = h3

    return p1, q1, h1


def bulk_discrete_log(base: int, values: Iterable[int], modulus: int) -> list[int]:
    """Computes dlog(base, v, modulus) for several values v.

    This can be faster than separate calls because analysis costs get amortized.
    For example, p-1 is only factored once instead of once per value.
    """

    # Canonicalize
    values = [v if 0 <= v < modulus else v % modulus for v in values]

    # Deduplicate.
    value_set = set(values)
    if len(value_set) < len(values):
        uniques = list(value_set)
        unique_solved = bulk_discrete_log(base, uniques, modulus)
        m = {u: s for u, s in zip(uniques, unique_solved)}
        return [m[v] for v in values]

    # Brute force small cases.
    if modulus < 1000:
        lookup = {}
        lookup[1] = 0
        lookup[base] = 1
        acc = base
        for k in range(2, modulus):
            acc *= base
            acc %= modulus
            lookup[acc] = k
        return [lookup[v] for v in values]

    # Fallback to cleverer things for big cases.
    factors = sympy.factorint(modulus - 1)
    if max(factors.values()) >= 100_000:
        return [sympy.discrete_log(modulus, v, base) for v in values]

    # Precompute lookups.
    lookups = []
    ms = []
    for prime, count in factors.items():
        period = prime**count
        m = (modulus - 1) // period
        ms.append(m)
        gm = pow(base, m, modulus)

        lookup = {}
        lookup[1] = 0
        lookup[gm] = 1
        acc = gm
        for k in range(2, period):
            acc *= gm
            acc %= modulus
            lookup[acc] = k
        lookups.append(lookup)

    # Use precomputed values to solve.
    results = []
    for v in values:
        cur_e = 0
        cur_v = 0
        for m, lookup in zip(ms, lookups):
            x, y, _ = gcd_ex(cur_v, m)
            cur_e = x * cur_e + y * m * lookup[pow(v, m, modulus)]
            cur_e %= modulus - 1
            cur_v = x * cur_v + y * m
            cur_v %= modulus - 1
        results.append(cur_e)

    return results


def table_str(
    entries: dict[complex, int | float | str],
    *,
    latex: bool = False,
    formatter: Callable[[int | float], str] | None = None,
) -> str:
    def expand_number(e: float | int | str) -> str:
        if formatter is not None:
            v = formatter(e)
            if v is not None:
                return str(v)
        if isinstance(e, float):
            if e == int(e):
                return str(int(e))
            return f"{e:.1f}"
        if isinstance(e, int) and not latex:
            r = str(e)
            r2 = ""
            for c in r[::-1]:
                if len(r2) % 4 == 3:
                    r2 += "_"
                r2 += c
            return r2[::-1]
        return str(e)

    entries = {k: expand_number(v) for k, v in entries.items()}

    min_row = min([int(k.imag) for k in entries.keys()], default=0)
    max_row = max([int(k.imag) for k in entries.keys()], default=0)

    col_widths = collections.defaultdict(int)
    for k, v in entries.items():
        col = int(k.real)
        col_widths[col] = max(col_widths[col], len(v))
    min_col = min(col_widths.keys(), default=0)
    max_col = max(col_widths.keys(), default=0)

    lines = []
    if latex:
        col_format = "|c" * (max_col - min_col + 1) + "|"
        lines.append(rf"""\begin{{tabular}}{{{col_format}}}""")
        lines.append(rf"""\hline""")
        for row in range(min_row, max_row + 1):
            is_seps = [
                entries.get(row * 1j + col, "") == "---" for col in range(min_col, max_col + 1)
            ]
            if any(is_seps):
                lines.append(rf"""\hline""")
            if all(is_seps):
                continue
            terms = []
            for col in range(min_col, max_col + 1):
                entry = entries.get(row * 1j + col, "")
                terms.append(rf"\text{{{entry}}}")
            lines.append(" & ".join(terms).rstrip())
            lines.append(rf"""\\""")
        lines.append(rf"""\hline""")
        lines.append(rf"""\end{{tabular}}""")
    else:
        for row in range(min_row, max_row + 1):
            terms = []
            for col in range(min_col, max_col + 1):
                entry = entries.get(row * 1j + col, "")
                pad = "-" if entry == "---" else " "
                entry = pad + entry.rjust(col_widths[col], pad) + pad
                terms.append(entry)
            lines.append("|".join(terms).rstrip())
    return "\n".join(lines)
