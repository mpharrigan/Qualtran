from __future__ import annotations

import argparse
import pathlib

from matplotlib import pyplot as plt


def generate_progression_plot(out_path: pathlib.Path | str | None, show: bool):
    entries = [
        # [2009, 6500e6, "M+2009"],  Omitted because cycle time 49x higher, error rate 2x higher
        [2010, 620e6, "Jones et al"],
        [2012, 1000e6, "Fowler et al"],
        [2017, 230e6, "O'Gorman et al"],
        [2019, 170e6, "Gheorghiu et al"],
        [2019, 20e6, "Gidney et al"],
        [2025, 1e6, "This Work"],
    ]
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    ax.scatter(
        [e[0] for e in entries],
        [e[1] for e in entries],
    )
    for x, y, t in entries:
        ax.plot([x], [y], color='C1' if x == 2025 else 'C0', marker='o', markersize=16)
        horizontalalignment = 'left'
        if t == "O'Gorman et al":
            horizontalalignment = 'right'
            t = t + "   "
        else:
            t = "   " + t
        ax.text(x, y, t, horizontalalignment=horizontalalignment, verticalalignment='top', fontsize=12, rotation=0)
    ax.semilogy()
    ax.set_xticks(range(2010, 2036, 5))
    ax.set_ylim(10**4, 2*10**9)
    ax.set_xlabel("Year", fontsize=16)
    ax.set_ylabel(rf"$\mathbf{{Physical}}$ Qubits", fontsize=16)
    ax.grid()
    fig.set_dpi(200)
    fig.set_size_inches(12, 7)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path)
        print(f'wrote file://{pathlib.Path(out_path).absolute()}')
    if out_path is None or show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    if args.out_path is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)
    generate_progression_plot(
        out_path=out_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
