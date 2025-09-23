import math


def approximate_modular_exponentiation(
        g: int,
        Q_e: int,
        N: int,
        P: list[int],
        f: int,
) -> int:
    """Computes an approximate modular exponentiation.

    Args:
        g: The base of the exponentiation.
        Q_e: The exponent.
        N: The modulus.
        P: Small primes for the residue arithmetic.
        f: Kept bits during approximate accumulation.

    Returns:
        An approximation of pow(g, Q_e, N).

        The modular deviation of the approximation is
        at most 3 * Q_e.bit_length() / 2**f .
    """
    L = math.prod(P)
    ell = max(p.bit_length() for p in P)
    m = Q_e.bit_length()
    t = N.bit_length() - f
    assert L == math.lcm(*P) and L >= N ** m and (L % N) < (N >> f)

    Q_total = 0
    for p in P:
        Q_residue = 1
        for k in range(Q_e.bit_length()):
            precomputed = pow(g, 1 << k, N) % p

            # controlled inplace modular multiplication:
            if Q_e & (1 << k):
                Q_residue *= precomputed
                Q_residue %= p

        u = (L // p) * pow(L // p, -1, p)
        for k in range(ell):
            precomputed = (((u << k) % L % N) >> t) % (N >> t)

            # controlled inplace modular addition:
            if Q_residue & (1 << k):
                Q_total += precomputed
                Q_total %= N >> t

    return Q_total << t


def test_approximate_modular_exponentiation():
    import random
    import sympy

    # RSA100 challenge number
    N = int(
        "15226050279225333605356183781326374297180681149613"
        "80688657908494580122963258952897654000350692006139"
    )
    g = random.randrange(2, N)
    Q_e = random.randrange(2**100)  # small exponent for quicker test
    P = [
        *sympy.primerange(239382, 2**18),
        131101,
        131111,
        131113,
        131129,
        131143,
        131149,
        131947,
        182341,
        239333,
        239347,
    ]
    f = 24
    result = approximate_modular_exponentiation(g, Q_e, N, P, f)
    error = result - pow(g, Q_e, N)
    deviation = min(error % N, -error % N) / N
    ell = max(p.bit_length() for p in P)
    assert deviation <= 3 * len(P) * ell / 2**f
