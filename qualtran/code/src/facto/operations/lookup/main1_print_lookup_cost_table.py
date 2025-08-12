from __future__ import annotations


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|c|c|}""")
    print(
        r"""        \hline n & Workspace &Multi-Target& Multi-Target   & Power-Product  & Reaction  & Execution Time"""
    )
    print(
        r"""        \\       & Qubits    &CX Gates (A)& CCX Gates (B)  & AND Gates (C)  & Depth (D) & (approximate)"""
    )
    R = 10
    S = 25
    M = 25
    for n in range(1, 11):
        m1 = n // 2
        m2 = n - m1
        p1 = (1 << m1) - 1
        p2 = (1 << m2) - 1
        power_product_and_gates = (p1 - m1) + (p2 - m2)
        multi_target_ccx_gates = p1 * p2
        multi_target_cx_gates = (1 << n) - 1 - multi_target_ccx_gates
        assert multi_target_cx_gates == p1 + p2
        ccz_states = power_product_and_gates + multi_target_ccx_gates
        assert ccz_states == (1 << n) - n - 1
        reactions = multi_target_ccx_gates + (n > 2)
        workspace = ((1 << m1) - m1 - 1) + ((1 << m2) - m2 - 1)
        execution_time = (
            1 * R
            + multi_target_ccx_gates * max(M, S, R)
            + multi_target_cx_gates * S
            + power_product_and_gates * M
        )
        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 6:
            execution_time = rf"\textbf{{{execution_time}}}"
            reactions = rf"\textbf{{{reactions}}}"
            multi_target_ccx_gates = rf"\textbf{{{multi_target_ccx_gates}}}"
            multi_target_cx_gates = rf"\textbf{{{multi_target_cx_gates}}}"
            power_product_and_gates = rf"\textbf{{{power_product_and_gates}}}"
            workspace = rf"\textbf{{{workspace}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} """
            rf"""& {workspace} """
            rf"""& {multi_target_cx_gates} """
            rf"""& {multi_target_ccx_gates} """
            rf"""& {power_product_and_gates} """
            rf"""& {reactions} """
            rf"""& {execution_time}"""
        )
    print(
        r"""        \\\hline n """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2^n - 2n$ """
        r"""& $\leq 2 \sqrt{2^n}$ """
        r"""& $\leq 2^n - 2n$ """
        r"""& $\begin{aligned} \approx"""
        r"& A \cdot S "
        r"\\&+ B \cdot \text{max}(M, R, S) "
        r"\\& + C \cdot M "
        r"\\& + R + nS"
        r"\end{aligned} $"
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
