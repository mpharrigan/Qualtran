from __future__ import annotations

import math


def main():
    print(r"""    \begin{tabular}{|c|c|c|c|c|}""")
    print(r"""        \hline n & Compute    & Uncompute  & CCZ States & Execution Time""")
    print(r"""        \\       & Layers (A) & Layers (B) & (C)        & (approximate)""")
    R = 10
    S = 25
    M = 25
    F = 6
    for n in range(5, 51, 5):
        compute_layers = math.ceil(n / F) * 6
        uncompute_layers = math.ceil(n / F) * 3
        ccz_states = n - 1
        execution_time = max(compute_layers * S, ccz_states * M, ccz_states * R) + max(
            uncompute_layers * S, ccz_states * R
        )
        execution_time = round(execution_time / 10) * 10
        execution_time = rf"{{{execution_time}}}us"
        if n == 35:
            compute_layers = rf"\textbf{{{compute_layers}}}"
            uncompute_layers = rf"\textbf{{{uncompute_layers}}}"
            ccz_states = rf"\textbf{{{ccz_states}}}"
            execution_time = rf"\textbf{{{execution_time}}}"
            n = rf"\textbf{{{n}}}"
        print(
            rf"""        \\\hline {n} """
            rf"""& {compute_layers} """
            rf"""& {uncompute_layers} """
            rf"""& {ccz_states} """
            rf"""& {execution_time}"""
        )
    print(
        r"""        \\\hline n """
        r"""& $6n / F$ """
        r"""& $2n / F$ """
        r"""& $n - 1$ """
        r"& $\begin{aligned} \approx &\text{max}(B \cdot S, C \cdot R) "
        r"\\&+ \text{max}(A \cdot S, C \cdot R, C \cdot M)"
        r"\end{aligned} $"
    )

    print(r"""        \\\hline""")
    print(r"""    \end{tabular}""")


if __name__ == "__main__":
    main()
