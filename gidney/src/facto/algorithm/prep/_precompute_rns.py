import dataclasses
import itertools
import math
import multiprocessing
import os
import queue
import sympy
import sys
import time
from collections.abc import Iterable
from typing import Any

from facto.algorithm.prep._problem_config import ProblemConfig
from facto.algorithm.prep._residue_util import iter_primes, choose


def prod_mod(values: tuple[int, ...], modulus: int) -> int:
    r = 1
    for v in values:
        r *= v
        r %= modulus
    return r


def prune(candidate: int, choice: tuple[int, ...], modulus: int) -> tuple[int, tuple[int, ...]]:
    s = set(choice)
    for p in choice:
        if candidate % p == 0:
            candidate //= p
            s.remove(p)
    return candidate, tuple(s)


def exec_worker(args: dict[str, Any]):
    choice_list: list[int] = list(args["choice_list"])
    contiguous_primes: list[int] = list(args["contiguous_primes"])
    worker_id: int = int(args["worker_id"])
    cpu_count: int = int(args["cpu_count"])
    modulus: int = int(args["modulus"])
    q: multiprocessing.Queue = args["q"]

    nb = modulus.bit_length()
    best_candidate = modulus
    for i, p1 in enumerate(choice_list[worker_id::cpu_count]):
        for p2 in choice_list[worker_id + i * cpu_count + 1 :]:
            choice = [p1, p2] + contiguous_primes
            candidate = prod_mod(choice, modulus)
            pruned_candidate, pruned_choice = prune(candidate, choice, modulus)
            if pruned_candidate < best_candidate:
                best_candidate = pruned_candidate
                q.put(("new_best_candidate", worker_id, pruned_candidate, pruned_choice))

    q.put(("done", worker_id))


@dataclasses.dataclass(frozen=True)
class FindRnsSolution:
    periods: tuple[int, ...]
    primes_bit_length: int | None
    primes_range_start: int | None
    primes_range_stop: int | None
    primes_extra: tuple[int, ...]
    primes_skipped: tuple[int, ...]


def _no_nil_mods(v) -> list[bool]:
    primes, not_nil_constraints = v
    return [all(f % prime for f in not_nil_constraints) for prime in primes]


def _verify_rns_solution(
    conf: ProblemConfig, solution: FindRnsSolution, *, print_progress: bool = False
) -> None:
    if print_progress:
        print("    verifying solution")

    primes_in_range = set(sympy.primerange(solution.primes_range_start, solution.primes_range_stop))
    expected_prime_count = (
        len(primes_in_range) + len(solution.primes_extra) - len(solution.primes_skipped)
    )
    actual_prime_count = len(solution.periods)
    assert (
        actual_prime_count == expected_prime_count
    ), f"Found {actual_prime_count} periods, expected {expected_prime_count}"

    for p in solution.primes_extra:
        assert p not in primes_in_range, f"Extra prime {p} in range"
    for p in solution.primes_skipped:
        assert p in primes_in_range, f"Skipped prime {p} not in range"

    L = math.prod(solution.periods)
    max_exponentiation_result = conf.modulus**conf.num_windows1
    assert L >= max_exponentiation_result, "Product of proposed periods is not large enough"

    for p in solution.periods:
        assert (
            p.bit_length() == solution.primes_bit_length
        ), f"Period {p} is not {solution.primes_bit_length} bit long"

    assert len(set(solution.periods)) == len(solution.periods), "Periods are not distinct"

    assert math.lcm(*solution.periods) == L, "Least common multiple of periods is too small"

    modular_deviation = L % conf.modulus
    if modular_deviation > conf.modulus // 2:
        modular_deviation = conf.modulus - modular_deviation

    shifted_modulus = conf.modulus >> conf.min_wraparound_gap
    assert (
        modular_deviation < shifted_modulus
    ), f"Modular deviation {modular_deviation} isn't smaller than shifted modulus {shifted_modulus}"

    if print_progress:
        print("    solution ok")


def find_rns_for_conf(
    conf: ProblemConfig, *, not_nil_constraints: Iterable[int] = (), print_progress: bool = False
) -> FindRnsSolution:
    """Finds prime periods suitable for truncated residue arithmetic."""
    not_nil_constraints = tuple(not_nil_constraints)
    max_product_bits = conf.modulus.bit_length() * conf.num_windows1
    if conf.rns_primes_bit_length is None:
        chosen_prime_bit_length = conf.estimate_minimum_rns_period_bit_length()
    else:
        chosen_prime_bit_length = conf.rns_primes_bit_length

    if print_progress:
        print("    sieving for primes...", file=sys.stderr)
    available_primes = []
    reached = set()
    for period in iter_primes():
        n = period.bit_length()
        if n > chosen_prime_bit_length:
            break
        if print_progress and n not in reached and n >= 15:
            reached.add(n)
            print(
                f"    reached {n} bit primes (target={chosen_prime_bit_length})...", file=sys.stderr
            )
        if period.bit_length() == chosen_prime_bit_length:
            available_primes.append(period)

    if not_nil_constraints:
        if print_progress:
            print("    discarding primes that are divisors of any multiplier...", file=sys.stderr)
        pool = multiprocessing.Pool()
        acceptables = pool.map(
            _no_nil_mods,
            [
                (available_primes[k : k + 1000], not_nil_constraints)
                for k in range(0, len(available_primes), 1000)
            ],
        )
        flat_acceptables = [e for a in acceptables for e in a]
        pool.close()
        p2acceptable = {p: a for p, a in zip(available_primes, flat_acceptables)}
    else:
        p2acceptable = {p: True for p in available_primes}

    if print_progress:
        print(
            "    brute forcing prime sets with small values of product(P) mod N...", file=sys.stderr
        )

    max_total = math.prod(available_primes)
    if max_total.bit_length() <= max_product_bits + chosen_prime_bit_length * 100:
        raise ValueError(
            f"rns_primes_bit_length={conf.rns_primes_bit_length} is set too low.\n"
            f"    The product of all {chosen_prime_bit_length} bit primes is a {max_total.bit_length()} bit number\n"
            f"    which doesn't leave enough slack to make random changes while hitting a target bit capacity of {max_product_bits}"
        )

    acceptable_primes = [p for p in available_primes if p2acceptable[p]]
    if print_progress:
        print(f"    found {len(acceptable_primes)} acceptable primes")
        max_exponentiation_result = conf.modulus**conf.num_windows1
        min_prime_count = 0
        product = 1
        for i, p in enumerate(reversed(acceptable_primes)):
            product *= p
            if product >= max_exponentiation_result:
                min_prime_count = i + 1
                break
        print(f"    need at least {min_prime_count} primes")

    biggest_prime = acceptable_primes.pop()
    second_biggest_prime = acceptable_primes.pop()
    fixed_total = acceptable_primes[0] * acceptable_primes[1]
    contiguous_primes = []
    while fixed_total.bit_length() <= max_product_bits:
        p = acceptable_primes.pop()
        contiguous_primes.append(p)
        fixed_total *= p
    contiguous_primes.append(biggest_prime)
    contiguous_primes.append(second_biggest_prime)

    if len(acceptable_primes) < 100:
        raise NotImplementedError(f"len(acceptable_primes) < 100")

    cpu_count = os.cpu_count()
    q = multiprocessing.Queue()
    workers = []
    for worker_id in range(cpu_count):
        worker = multiprocessing.Process(
            target=exec_worker,
            args=(
                {
                    "choice_list": acceptable_primes,
                    "contiguous_primes": contiguous_primes,
                    "worker_id": worker_id,
                    "cpu_count": cpu_count,
                    "modulus": conf.modulus,
                    "q": q,
                },
            ),
        )
        worker.start()
        workers.append(worker)
    t0 = time.monotonic()
    finished = 0
    shifted_modulus = conf.modulus >> conf.min_wraparound_gap
    best_candidate = conf.modulus
    best_choice: tuple[int, ...] = ()
    while best_candidate >= shifted_modulus:
        try:
            v = q.get(timeout=1)
            if v[0] == "done":
                finished += 1
                if finished == len(workers):
                    raise NotImplementedError("Search space exhausted.")
            elif v[0] == "new_best_candidate":
                _, worker_id, candidate, choice = v
                if candidate < best_candidate:
                    best_candidate = candidate
                    best_choice = tuple(choice)
                    if print_progress:
                        shift = conf.modulus.bit_length() - candidate.bit_length()
                        print(
                            f"    best so far: N >> {shift} (target=N>>{conf.min_wraparound_gap}, elapsed={math.ceil(time.monotonic() - t0)}s)",
                            file=sys.stderr,
                        )
            else:
                raise NotImplemented(f"{v=}")
        except queue.Empty:
            pass
    assert best_choice
    for w in workers:
        w.terminate()
    for w in workers:
        w.join()
    for w in workers:
        w.close()
    workers.clear()
    q.close()

    if print_progress:
        print(f"    found solution consisting of {len(best_choice)} primes")

    primes_range_start = min(contiguous_primes) - 1
    primes_range_stop = max(contiguous_primes) + 1
    primes_extra = []
    for p in best_choice:
        if p < primes_range_start or primes_range_stop < p:
            primes_extra.append(p)
    primes_skipped = []
    best_choice_set = set(best_choice)
    for p in available_primes:
        if p < primes_range_start or primes_range_stop < p:
            continue
        if p not in best_choice_set:
            primes_skipped.append(p)

    rns_solution = FindRnsSolution(
        periods=tuple(sorted(best_choice)),
        primes_bit_length=chosen_prime_bit_length,
        primes_range_start=min(contiguous_primes) - 1,
        primes_range_stop=max(contiguous_primes) + 1,
        primes_extra=tuple(sorted(primes_extra)),
        primes_skipped=tuple(sorted(primes_skipped)),
    )
    _verify_rns_solution(conf, rns_solution, print_progress=print_progress)
    return rns_solution
