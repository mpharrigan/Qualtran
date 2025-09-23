import numpy as np

from facto.algorithm.prep import ExecutionConfig
from scatter_script import QPU, quint, Lookup


def approx_modexp(
        Q_exponent: quint,
        conf: ExecutionConfig,
        qpu: QPU,
) -> quint:
    """Quantum computes an approximate modular exponentation.

    Args:
        Q_exponent: The superposed exponent to exponentiate by.
        conf: Problem configuration data, precomputed tables, etc.
        qpu: Simulator instance being acted upon.

    Returns:
        The result of the approximate modular exponentation.
    """

    Q_dlog: quint = qpu.alloc_quint(length=conf.len_dlog_accumulator)
    Q_result: quint = qpu.alloc_quint(
        length=conf.len_accumulator + 1,
        scatter=True,  # Initialized to a superposition mask.
        scatter_range=1 << conf.mask_bits,
    )
    loop1_vent = Lookup(np.zeros((conf.num_windows1, 1 << conf.window1), dtype=np.bool_))

    for i in range(len(conf.periods)):
        p = int(conf.periods[i])

        # Offset Q_dlog to equal dlog(generators[i], residue, p)
        loop1(
            Q_dlog=Q_dlog,
            Q_exponent=Q_exponent,
            vent=loop1_vent,
            i=i,
            conf=conf,
        )

        # Compress Q_dlog while preserving Q_dlog % (p - 1)
        Q_compressed = loop2(
            Q_target=Q_dlog,
            modulus=p - 1,
            compressed_len=p.bit_length(),
        )

        # Perform: let Q_residue := pow(generators[i], Q_dlog, p)
        Q_residue = loop3(
            Q_dlog=Q_compressed,
            i=i,
            conf=conf,
            qpu=qpu,
        )

        # Approximate: Q_result += Q_residue * (L//p) * pow(L//p, -1, p)
        loop4(
            Q_residue=Q_residue,
            Q_acc=Q_result,
            i=i,
            conf=conf,
            qpu=qpu,
        )

        # Perform: del Q_residue := pow(generators[i], Q_dlog, p)
        unloop3(
            Q_unresult=Q_residue,
            Q_dlog=Q_compressed,
            i=i,
            conf=conf,
            qpu=qpu,
        )

        # Uncompress Q_dlog
        unloop2(
            Q_target=Q_dlog,
            modulus=p - 1,
            compressed_len=p.bit_length(),
        )

    # Uncompute Q_dlog.
    loop1(
        Q_dlog=Q_dlog,
        Q_exponent=Q_exponent,
        vent=loop1_vent,
        i=len(conf.periods),
        conf=conf,
    )
    Q_dlog.del_by_equal_to(0)

    # Clear accumulated phase corrections from Q_dlog-related lookups.
    for j in range(conf.num_windows1):
        Q_k = Q_exponent[j * conf.window1 :][: conf.window1]
        qpu.z(loop1_vent[j, Q_k])

    return Q_result


def loop1(
        Q_dlog: quint,
        conf: ExecutionConfig,
        Q_exponent: quint,
        i: int,
        vent: Lookup,
) -> None:
    """Offsets Q_dlog to the discrete log of the next residue.

    Args:
        Q_dlog: The register to offset.
        conf: Specifies details like the base of the exponent for the current
            residue, window sizes, and precomputed tables.
        Q_exponent: In the problem spec this is the value to exponentiate by,
            but in this method it's the value to use as addresses for table
            lookups that get added together.
        i: The iteration variable of the loop iterating over the primes in
            the residue number system. Indexes the modulus and generator,
            as well as related table data, from within `conf`.
        vent: Where to dump deferred phase data for uncomputing lookups.
    """
    for j in range(conf.num_windows1):
        Q_k = Q_exponent[j * conf.window1 :][: conf.window1]
        table = conf.lookup1[i, j].venting_into(vent[j])
        Q_dlog += table[Q_k]


def loop2(
        Q_target: quint,
        modulus: int,
        compressed_len: int,
) -> quint:
    """Compresses `Q_target % modulus` into a smaller part of `Q_target`.

    Args:
        Q_target: The register containing the remainder to compress.
        compressed_len: How large the remainder should be
            after compression. At least modulus.bit_length().
        modulus: The modulus used in the remainder computation.

    Returns:
        The slice of `Q_target` that the remainder is in.
    """
    n = len(Q_target)
    while n > compressed_len:
        n -= 1
        threshold = modulus << (n - compressed_len)

        # Perform a step of binary long division.
        Q_target[: n + 1] -= threshold
        Q_target[:n] += Q_target[n].ghz_lookup(threshold)

    return Q_target[:n]


def loop3(
        conf: ExecutionConfig,
        Q_dlog: quint,
        i: int,
        qpu: QPU,
) -> quint:
    """Computes `pow(conf.generators[i], Q_exponent, conf.periods[i])`.

    Args:
        conf: Specifies details like the base of the exponent for the current
            residue, window sizes, and precomputed tables.
        Q_dlog: Superposed value to exponentiate by.
        i: The iteration variable of the loop iterating over the primes in
            the residue number system. Indexes the modulus and generator,
            as well as related table data, from within `conf`.
        qpu: Simulator instance being acted upon.

    Returns:
        An allocated register containing the result of the exponentation.
    """
    modulus = int(conf.periods[i])
    Q_result = qpu.alloc_quint(length=modulus.bit_length() + 1)
    Q_helper = qpu.alloc_quint(length=modulus.bit_length() + 1)

    # Skip first 2 iterations by direct lookup.
    table = conf.lookup3c[i].venting_into_new_table()
    Q_l = Q_dlog[: conf.window3a * 2]
    Q_result ^= table[Q_l]
    qpu.push_uncompute_info(table.vent)

    # Decompose exponentiation into windowed multiplications.
    for j in range(2, conf.num_windows3a):
        Q_l1 = Q_dlog[j * conf.window3a :][: conf.window3a]

        # Perform `let Q_helper := Q_residue * X % N`
        for k in range(conf.num_windows3b):
            Q_l0 = Q_result[k * conf.window3b :][: conf.window3b]
            Q_l = (Q_l1 << conf.window3b) | Q_l0
            table = conf.lookup3a[i, j, k].venting_into_new_table()

            # Subtraction mod the modulus.
            Q_helper -= table[Q_l]
            Q_helper[:-1] += Q_helper[-1].ghz_lookup(modulus)

            # Defer phase corrections to `unloop3` method.
            phase_wrap = Q_helper[-1].mx_rz()
            qpu.push_uncompute_info((phase_wrap, table.vent))

        Q_result, Q_helper = Q_helper, Q_result

        # Defer `del Q_helper := Q_residue * X^-1 % N` to `unloop3`
        qpu.push_uncompute_info(Q_helper.mx_rz())

    Q_helper.del_by_equal_to(0)
    return Q_result


def loop4(
        conf: ExecutionConfig,
        Q_residue: quint,
        Q_acc: quint,
        i: int,
        qpu: QPU,
) -> None:
    """Adds Q_residue's approximate contribution into Q_result_accumulator.

    Args:
        conf: Specifies details like the base of the exponent for the current
            residue, window sizes, and precomputed tables.
        Q_residue: The residue of the exact total for the current modulus.
        Q_acc: Where to add the residue's approximate contributions.
        i: The iteration variable of the loop iterating over the primes in
            the residue number system. Indexes the modulus and generator,
            as well as related table data, from within `conf`.
        qpu: Simulator instance being acted upon.
    """
    trunc = conf.truncated_modulus
    for j in range(conf.num_windows4):
        Q_k = Q_residue[j * conf.window4 :][: conf.window4]
        table = conf.lookup4[i, j].venting_into_new_table()
        table2 = trunc - table

        # Subtraction mod the truncated modulus.
        Q_acc -= table[Q_k]
        Q_acc[:-1] += Q_acc[-1].ghz_lookup(trunc)

        if Q_acc[-1].mx_rz():
            # Fix wraparound phase with a comparison.
            qpu.z(Q_acc[:-1] >= table2[Q_k])

        # Fix deferred corrections with a phase lookup.
        qpu.z(table.vent[Q_k])


def unloop3(
        Q_unresult: quint,
        conf: ExecutionConfig,
        Q_dlog: quint,
        i: int,
        qpu: QPU,
) -> None:
    """Uncomputes `pow(conf.generators[i], Q_exponent, conf.periods[i])`.

    Args:
        Q_unresult: The register to uncompute and delete.
        conf: Specifies details like the base of the exponent for the current
            residue, window sizes, and precomputed tables.
        Q_dlog: Superposed value to exponentiate by.
        i: The iteration variable of the loop iterating over the primes in
            the residue number system. Indexes the modulus and generator,
            as well as related table data, from within `conf`.
        qpu: Simulator instance being acted upon.
    """
    modulus = int(conf.periods[i])
    Q_helper = qpu.alloc_quint(length=modulus.bit_length() + 1)

    # Decompose un-exponentiation into windowed un-multiplications.
    for j in range(2, conf.num_windows3a)[::-1]:
        Q_l0 = Q_dlog[j * conf.window3a :][: conf.window3a]

        # Perform `let Q_helper := Q_unresult * X^-1 % N`
        for k in range(conf.num_windows3b)[::-1]:
            Q_l1 = Q_unresult[k * conf.window3b :][: conf.window3b]
            Q_l = (Q_l0 << conf.window3b) | Q_l1
            table1 = conf.lookup3b[i, j, k].venting_into_new_table()
            table2 = modulus - table1

            # Subtraction mod the modulus.
            Q_helper -= table2[Q_l]
            Q_helper[:-1] += Q_helper[-1].ghz_lookup(modulus)

            # Uncompute wraparound qubit.
            not_phase_wrap = Q_helper[-1].mx_rz()
            qpu.z(not_phase_wrap)  # Only for phase cleared verification.
            if not_phase_wrap:
                # Fix wraparound phase with a comparison.
                qpu.z(Q_helper[:-1] < table1[Q_l])

            # Fix deferred corrections with a phase lookup.
            qpu.z(table1.vent[Q_l])

        phase_mask_from_helper_during_compute = qpu.pop_uncompute_info()
        qpu.cz(Q_helper, phase_mask_from_helper_during_compute)
        Q_unresult, Q_helper = Q_helper, Q_unresult

        # Perform `del Q_helper := Q_unresult * X % N`
        for k in range(conf.num_windows3b)[::-1]:
            Q_l1 = Q_unresult[k * conf.window3b :][: conf.window3b]
            Q_l = (Q_l0 << conf.window3b) | Q_l1
            phase_wrap, phase_table = qpu.pop_uncompute_info()
            table1 = conf.lookup3a[i, j, k].venting_into(phase_table)
            table2 = modulus - table1

            # Subtraction mod the modulus.
            Q_helper -= table2[Q_l]
            Q_helper[:-1] += Q_helper[-1].ghz_lookup(modulus)

            # Uncompute wraparound qubit.
            not_phase_wrap = Q_helper[-1].mx_rz()
            qpu.z(not_phase_wrap)  # Only for phase cleared verification.
            if phase_wrap ^ not_phase_wrap:
                # Fix wraparound phase with a comparison.
                qpu.z(Q_helper[:-1] < table1[Q_l])

            # Fix deferred corrections with a phase lookup.
            qpu.z(phase_table[Q_l])

    # Skip last 2 iterations by measurement based uncomputation of lookup.
    Q_helper.del_by_equal_to(0)
    mx = Q_unresult.del_measure_x()
    vent = qpu.pop_uncompute_info()
    vent ^= conf.lookup3c[i].phase_corrections_for_mx(mx)
    Q_l = Q_dlog[: conf.window3a * 2]
    qpu.z(vent[Q_l])


def unloop2(
        Q_target: quint,
        modulus: int,
        compressed_len: int,
) -> None:
    """Uncompresses `Q_target % modulus` from a smaller part of `Q_target`.

    Args:
        Q_target: The register containing the remainder to uncompress.
        compressed_len: How large the remainder was
            after compression. At least modulus.bit_length().
        modulus: The modulus used in the remainder computation.
    """

    n = compressed_len
    while n < len(Q_target):
        threshold = modulus << (n - compressed_len)

        # Un-perform a step of binary long division.
        Q_target[:n] -= Q_target[n].ghz_lookup(threshold)
        Q_target[: n + 1] += threshold

        n += 1
