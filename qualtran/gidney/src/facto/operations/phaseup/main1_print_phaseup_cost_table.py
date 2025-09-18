from __future__ import annotations

from facto.operations.phaseup._phaseup_imp import estimate_cost_of_sqrt_phaseup
from scatter_script import CostKey


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|c|}""")
    print(
        r"""        \hline n & Workspace   & Wandering    & Multi-Target  & Reaction  & Execution Time"""
    )
    print(
        r"""        \\       & Qubits      & AND Gates    & CZ Gates      & Depth     & (approximate)"""
    )
    R = 10
    S = 25
    M = 25
    for n in range(1, 11):
        e = estimate_cost_of_sqrt_phaseup(n)
        assert len(e) == 2
        multi_czs = e[CostKey("CZ_multi_target")]
        ands = e[CostKey("uncorrected_and")]
        reactions = (n + 3) // 2
        m1 = n // 2
        m2 = n - m1
        workspace = ((1 << m1) - m1 - 1) + ((1 << m2) - m2 - 1)
        if n < 2:
            reactions = 0
        execution_time = reactions * R + (multi_czs + n) * S + ands * M
        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 6:
            execution_time = rf"\textbf{{{execution_time}}}"
            reactions = rf"\textbf{{{reactions}}}"
            multi_czs = rf"\textbf{{{multi_czs}}}"
            ands = rf"\textbf{{{ands}}}"
            workspace = rf"\textbf{{{workspace}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} & {workspace} & {ands} & {multi_czs} & {reactions} & {execution_time}"""
        )
    print(
        r"""        \\\hline n & $\leq 2 \sqrt{2^n}$ & $\leq 2 \sqrt{2^n}$ & $\leq \sqrt{2^n}$ & $(n + 3) // 2$ & $\begin{aligned} \approx& \text{AND}\cdot M \\&+ (\text{MultiCZ} + n) \cdot S \\&+\text{Reaction}\cdot R \end{aligned} $"""
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
