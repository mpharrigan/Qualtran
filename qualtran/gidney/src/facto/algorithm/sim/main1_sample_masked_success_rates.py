import argparse
import collections
import fractions
import itertools
import math
import multiprocessing
import pathlib
import random
import sys

import numpy as np
import sympy


def compute_periodic_signal(g: int, modulus: int) -> np.ndarray:
    n = modulus.bit_length()
    acc = np.zeros(1 << n, dtype=np.uint64)
    acc[0] = 1
    for k in range(n):
        mult = pow(g, 2**k, modulus)
        m = 1 << k
        view = acc[m:2*m]
        view ^= acc[:m]
        view *= mult
        view %= modulus
        if np.count_nonzero(view == 1):
            c = np.where(view == 1)[0][0]
            return acc[:m + c]


class C:
    def __init__(self, modulus: int, g: int, use_randomization: bool):
        self.g = g
        self.modulus = modulus
        self.use_randomization = use_randomization

        # Brute force the periodic signal produced by the given choice of (g, modulus).
        self.signal = compute_periodic_signal(g, modulus)
        self.period = len(self.signal)

        # Precompute quantum results that will cause postprocessing to succeed.
        self.success_mask = np.zeros(shape=self.period, dtype=np.bool_)
        for k in range(self.period):
            d = fractions.Fraction(k, self.period).limit_denominator(modulus).denominator
            f = math.gcd(pow(g, d // 2, modulus) + 1, modulus)
            self.success_mask[k] |= 1 < f < modulus

    def sample_chance_of_successful_factor(
            self,
            mask_width: int,
    ) -> float:
        # Sample from (uniform(mask_width) + g**uniform(period)) mod modulus
        measured = int(self.signal[random.randrange(self.period)])
        random_factor = 1
        if self.use_randomization:
            while True:
                random_factor = random.randrange(2, self.modulus - 2)
                if math.gcd(random_factor, self.modulus) == 1:
                    break
        measured *= random_factor
        measured += random.randrange(mask_width)
        measured %= self.modulus

        # Determine the set of kept signal values.
        cutoff = measured - self.signal.astype(np.int64) * random_factor
        cutoff += self.modulus
        cutoff %= self.modulus
        kept = cutoff < mask_width
        amps = kept.astype(np.float32) * math.sqrt(1.0 / np.sum(kept))
        freq_amps = np.fft.fft(amps, norm='ortho')
        dist2 = np.abs(freq_amps)**2

        return np.sum(dist2 * self.success_mask)


def sample_semi_prime(tup: tuple[int, dict[float, int], set[float], bool]) -> tuple[int, dict[float, int], dict[float, float]]:
    modulus: int
    shots_left: dict[float, int]
    mask_proportions: set[float]
    use_randomization: bool
    modulus, shots_left, mask_proportions, use_randomization = tup
    shots_left = dict(shots_left)
    hits = {p: 0 for p in mask_proportions}
    shots_taken = {p: 0 for p in mask_proportions}
    while True:
        g = modulus
        while math.gcd(g, modulus) != 1:
            g = random.randrange(2, modulus - 2)

        c = C(modulus=modulus, g=g, use_randomization=use_randomization)
        did_work = False
        for p in shots_left.keys():
            if shots_left[p] > 0:
                did_work = True
                mask_width = max(1, round(modulus * p))
                shots_taken[p] += 1
                hits[p] += c.sample_chance_of_successful_factor(mask_width=mask_width)
                shots_left[p] -= 1
        if not did_work:
            break
    return modulus, shots_taken, hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_semiprime', default=0, type=int)
    parser.add_argument('--semiprimes', default=(), type=int, nargs='+')
    parser.add_argument('--max_shots', required=True, type=int)
    parser.add_argument('--mask_proportions', required=True, type=float, nargs='+')
    parser.add_argument('--use_randomization', action='store_true')
    parser.add_argument('--save_resume_path', required=True, type=str)
    parser.add_argument('--log', action='store_true')
    args = parser.parse_args()

    max_semiprime = args.max_semiprime
    mask_proportions = set(args.mask_proportions)
    max_shots = args.max_shots
    save_resume_path = args.save_resume_path
    use_randomization = args.use_randomization

    p2m2shots: collections.defaultdict[float, collections.defaultdict[int, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    if pathlib.Path(save_resume_path).exists():
        with open(save_resume_path) as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith('modulus,'):
                    continue
                pieces = line.split(',')
                modulus, mask_proportion, shots, successes = pieces
                modulus = int(modulus)
                shots = int(shots)
                mask_proportion = float(mask_proportion)
                p2m2shots[mask_proportion][modulus] += shots
    else:
        with open(save_resume_path, 'w') as f:
            print("modulus,mask_proportion,shots,successes", file=f)

    primes = sympy.primerange(3, max_semiprime)
    semi_primes = sorted(
        a*b
        for a, b in itertools.combinations(primes, 2)
    )
    semi_primes = [e for e in semi_primes if e < max_semiprime]
    semi_primes.extend(args.semiprimes)


    pool = multiprocessing.Pool()
    modulus: int
    shots_taken: dict[float, float]
    hits: dict[float, int]
    tups = []
    for s in semi_primes:
        tups.append((s, {p: max_shots - p2m2shots[p][s] for p in mask_proportions}, mask_proportions, use_randomization))
    with open(save_resume_path, 'a') as f:
        if args.log:
            print("modulus,mask_proportion,shots,successes")
        for modulus, shots_taken, hits in pool.imap_unordered(sample_semi_prime, tups):
            for p, s in shots_taken.items():
                if s:
                    h = hits[p]
                    if args.log:
                        print(f'{modulus},{p},{s},{h}')
                    print(f'{modulus},{p},{s},{h}', file=f, flush=True)
    print(f"finished updating file://{pathlib.Path(save_resume_path).absolute()}", file=sys.stderr)


if __name__ == '__main__':
    main()
