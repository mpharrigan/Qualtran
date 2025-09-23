import argparse
import collections
import math
import pathlib
import random
from typing import Sequence, Callable

import numpy as np

import gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default=None, type=str)
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--g', required=True, type=int)
    parser.add_argument('--mask_proportion', required=True, type=float)
    args = parser.parse_args()

    if args.out_path is None:
        out_path = None
    else:
        out_path = pathlib.Path(args.out_path)
        out_path.parent.mkdir(exist_ok=True, parents=True)

    modulus = args.n
    mask_width = round(modulus * args.mask_proportion)
    g = args.g

    content = "\n".join(do_plot(mask_width=mask_width, modulus=modulus, g=g, exponent_range=None))
    svg = gen.str_svg(content)
    if out_path is None:
        print(svg)
    else:
        svg.write_to(out_path)


def do_plot(*, mask_width: int, modulus: int, g: int, exponent_range: int | None) -> list[str]:
    period = 1
    g2 = g
    while g2 != 1:
        g2 *= g
        g2 %= modulus
        period += 1
    if exponent_range is None:
        exponent_range = period
    w = exponent_range
    sec = 300

    lines = []
    lines.append(
        f"""<svg
        viewBox="{0} {0} {sec*3} {sec*2}"
        xmlns="http://www.w3.org/2000/svg"
    >"""
    )

    plot_top_left, plot_span = perimeter_label_cut(
        top_left=0,
        span=sec * (1 + 2j),
        top_label="Pre-measurement Superposition",
        left_label=f"Exponent Register (e) (mod Period={period})",
        bottom_label=f"Value Register: V = (uniform({mask_width}) + {g}^e) mod {modulus}",
        out=lines,
    )
    plot_top_left, plot_span, x2show, y2show = tick_labels_cut(
        top_left=plot_top_left,
        span=plot_span,
        min_x=0,
        max_x=modulus,
        min_y=0,
        max_y=period,
        out=lines,
        log_y=False,
        show_grid_lines=False,
    )
    lines.append(
        f"""<rect
        x="{plot_top_left.real}"
        y="{plot_top_left.imag}"
        width="{plot_span.real}"
        height="{plot_span.imag}"
        stroke="black"
        fill="none"
        stroke-width="0.5"
    />"""
    )

    total_averaged_fft = np.zeros(w, dtype=np.float64)
    total_matches = 0
    m2hits = collections.Counter()
    amps = collections.Counter()
    for x in range(w):
        offset_y = pow(g, x, modulus)
        offset_y %= modulus
        x2 = (offset_y + 0) % modulus
        x3 = (offset_y + mask_width) % modulus
        if x3 < x2:
            xxs = [(x2, modulus), (0, x3 % modulus)]
        else:
            xxs = [(x2, x3)]
        for x2, x3 in xxs:
            x2v = x2show(x2)
            x3v = x2show(x3)
            lines.append(
                f"""<rect
                fill="black"
                stroke="none"
                x="{x2v}"
                y="{y2show(x)}"
                width="{x3v - x2v}"
                height="1"
            />"""
            )
        for dy in range(mask_width):
            y2 = (offset_y + dy) % modulus
            m2hits[y2] += 1
            amps[x + y2 * 1j] += 1

    num_non_zero_amplitudes = sum(m2hits.values())
    svg_plot(
        top_left=sec * (1 + 0j),
        span=sec * (2 + 1j),
        ys=[m2hits[x] / num_non_zero_amplitudes for x in range(modulus)],
        min_y=1e-5,
        max_y=1e-1,
        out=lines,
        fill="#FF0000",
        stroke="#FF0000",
        title=f"Distribution of V",
        y_label="Probability",
        x_label="Measurement Result",
    )

    sample_driver = random.randrange(sum(m2hits.values()))
    for sampled_measurement, v in m2hits.items():
        sample_driver -= v
        if sample_driver <= 0:
            break
    else:
        assert False

    sampled_ys = np.zeros(w, dtype=np.float64)
    for x in range(w):
        sampled_ys[x] = amps[x + 1j * sampled_measurement]
    sampled_ys /= np.linalg.norm(sampled_ys)

    averaged_fft = np.zeros(w, dtype=np.float64)
    for possible_measurement in range(modulus):
        amps_col = np.zeros(w, dtype=np.float64)
        for x in range(w):
            amps_col[x] = amps[x + 1j * possible_measurement]
        averaged_fft += np.abs(np.fft.fft(amps_col, norm="ortho")) ** 2
    averaged_fft /= np.sum(averaged_fft)
    svg_plot(
        top_left=sec * (1 + 1j),
        span=sec * (2 + 1j),
        ys=averaged_fft,
        min_y=1e-5,
        max_y=1e-0,
        out=lines,
        fill="#0000FF",
        stroke="#0000FF",
        title=f"Distribution of QFT(e)",
        y_label="Probability",
        x_label=f"Frequency (mod Period={period})",
    )
    total_averaged_fft += averaged_fft
    total_matches += 1
    lines.append("""</svg>""")
    return lines


def choose_ticks_log(min_y: float, max_y: float) -> list[tuple[str, float]]:
    y_tick_min = math.floor(math.log10(min_y)) - 1
    y_tick_max = math.ceil(math.log10(max_y)) + 1
    y_ticks = [
        (f"1e{y_tick}", 10**y_tick)
        for y_tick in range(y_tick_min, y_tick_max + 1)
        if min_y <= 10**y_tick <= max_y
    ]
    return y_ticks


def choose_ticks_linear(min_x: float, max_x: float) -> list[tuple[str, float]]:
    x_tick_min = math.floor(min_x)
    x_tick_max = math.ceil(max_x)
    base = 10 ** (math.floor(math.log10(x_tick_max - x_tick_min)) - 3)
    choices = [base * k * 10**b for k in [1, 2, 2.5, 5] for b in [0, 1, 2, 3, 4]]
    for c in choices:
        c = int(c)
        if c == 0:
            continue
        x_tick_step = c
        r = range(x_tick_min, x_tick_max + 1, x_tick_step)
        if 5 <= len(r) <= 30:
            x_ticks = list(r)
            break
    else:
        assert False
    if x_ticks[-1] > x_tick_max - x_tick_step / 2:
        x_ticks.pop()
    x_ticks.append(x_tick_max)
    return [(str(e), e) for e in x_ticks]


def tick_labels_cut(
    *,
    top_left: complex,
    span: complex,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    log_y: bool,
    out: list[str],
    show_grid_lines: bool = True,
) -> tuple[complex, complex, Callable[[float], float], Callable[[float], float]]:
    inner_top_left = top_left + 25
    inner_span = span - 25 - 25j
    inner_span -= 4  # Prevent the rightmost tick label from clipping outside the area

    uw = inner_span.real / (max_x - min_x)

    def x2show(x_val: float) -> float:
        return x_val * uw + inner_top_left.real

    if log_y:
        min_plotted_y = math.log(min_y)
        max_plotted_y = math.log(max_y)

        def y2show(y_val: float) -> float:
            if y_val <= min_y:
                return inner_top_left.imag + inner_span.imag
            plotted_y = math.log(y_val)
            normalized_y = (plotted_y - min_plotted_y) / (max_plotted_y - min_plotted_y)
            return inner_top_left.imag + inner_span.imag - normalized_y * inner_span.imag

    else:
        uh = inner_span.imag / (max_y - min_y)

        def y2show(y_val: float) -> float:
            return -y_val * uh + inner_top_left.imag + inner_span.imag

    if log_y:
        y_ticks = choose_ticks_log(min_y, max_y)
    else:
        y_ticks = choose_ticks_linear(min_y, max_y)
    for label, y_tick_val in y_ticks:
        show_x1 = inner_top_left.real - 5
        if show_grid_lines:
            show_x2 = inner_top_left.real + inner_span.real
        else:
            show_x2 = inner_top_left.real
        show_y = y2show(y_tick_val)
        v_anchor = "central"
        out.append(
            f"""<text
            x="{show_x1}"
            y="{show_y}"
            fill="black"
            font-size="10"
            text-anchor="end"
            dominant-baseline="{v_anchor}"
        >
            {label}
        </text>"""
        )
        out.append(
            f"""<path
            d="M{show_x1},{show_y} L{show_x2},{show_y}"
            stroke="gray"
            fill="none"
            stroke-width="0.5"
        />"""
        )

    x_ticks = choose_ticks_linear(min_x, max_x)
    for label, x_tick in x_ticks:
        if show_grid_lines:
            show_y1 = inner_top_left.imag
        else:
            show_y1 = inner_top_left.imag + inner_span.imag
        show_y2 = inner_top_left.imag + inner_span.imag + 5
        show_x = x2show(x_tick)
        out.append(
            f"""<text
            x="0"
            y="0"
            fill="black"
            font-size="{10}"
            text-anchor="end"
            dominant-baseline="middle"
            transform="translate({show_x}, {show_y2}) rotate(-90)"
        >
            {label}
        </text>"""
        )
        out.append(
            f"""<path
            d="M{show_x},{show_y1} L{show_x},{show_y2}"
            stroke="gray"
            fill="none"
            stroke-width="0.5"
        />"""
        )

    return inner_top_left, inner_span, x2show, y2show


def perimeter_label_cut(
    *,
    top_left: complex,
    span: complex,
    top_label: str | None = None,
    left_label: str | None = None,
    right_label: str | None = None,
    bottom_label: str | None = None,
    out: list[str],
) -> tuple[complex, complex]:
    font_size = 10
    font_height = font_size + 5
    inner_top_left = top_left
    inner_span = span
    if top_label:
        inner_top_left += 1j * font_height
        inner_span -= 1j * font_height
    if bottom_label:
        inner_span -= 1j * font_height
    if left_label:
        inner_top_left += font_height
        inner_span -= font_height
    if right_label:
        inner_span -= font_height

    if top_label:
        out.append(
            f"""<text
            x="{inner_top_left.real + inner_span.real / 2}"
            y="{(top_left.imag + inner_top_left.imag) / 2}"
            fill="black"
            font-size="{font_size}"
            text-anchor="middle"
            dominant-baseline="middle"
        >
            {top_label}
        </text>"""
        )
    if bottom_label:
        out.append(
            f"""<text
            x="{inner_top_left.real + inner_span.real / 2}"
            y="{(top_left.imag + span.imag + inner_top_left.imag + inner_span.imag) / 2}"
            fill="black"
            font-size="{font_size}"
            text-anchor="middle"
            dominant-baseline="middle"
        >
            {bottom_label}
        </text>"""
        )
    if left_label:
        cx = (top_left.real + inner_top_left.real) / 2
        cy = inner_top_left.imag + inner_span.imag / 2
        out.append(
            f"""<text
            x="0"
            y="0"
            fill="black"
            font-size="{font_size}"
            text-anchor="middle"
            dominant-baseline="middle"
            transform="translate({cx}, {cy}) rotate(-90)"
        >
            {left_label}
        </text>"""
        )
    if right_label:
        cx = (top_left.real + span.real + inner_top_left.real + inner_span.real) / 2
        cy = inner_top_left.imag + inner_span.imag / 2
        out.append(
            f"""<text
            x="0"
            y="0"
            fill="black"
            font-size="{font_size}"
            text-anchor="middle"
            dominant-baseline="middle"
            transform="translate({cx}, {cy}) rotate(-90)"
        >
            {left_label}
        </text>"""
        )
    return inner_top_left, inner_span


def svg_plot(
    *,
    top_left: complex,
    span: complex,
    ys: Sequence[float],
    ys2: Sequence[float] | None = None,
    fill2: str | None = None,
    min_y: float,
    max_y: float,
    out: list[str],
    fill: str,
    stroke: str,
    stroke2: str | None = None,
    title: str,
    x_label: str,
    y_label: str,
):
    top_left, span = perimeter_label_cut(
        top_left=top_left,
        span=span,
        top_label=title,
        left_label=y_label,
        bottom_label=x_label,
        right_label=None,
        out=out,
    )
    top_left, span, x2show, y2show = tick_labels_cut(
        top_left=top_left,
        span=span,
        min_y=min_y,
        max_y=max_y,
        min_x=0,
        max_x=len(ys),
        out=out,
        log_y=True,
    )

    bottom_left = top_left + span.imag * 1j

    for ys_x, fill_x, stroke_x in [(ys, fill, stroke), (ys2, fill2, stroke2)]:
        if ys_x is None or fill_x is None:
            continue
        prev_show_x = bottom_left.real
        dirs = [f"M{prev_show_x},{bottom_left.imag}"]
        for x in range(len(ys)):
            show_x = x2show(x + 1)
            show_y = y2show(ys_x[x])
            dirs.append(f"L{prev_show_x},{show_y}")
            dirs.append(f"L{show_x},{show_y}")
            prev_show_x = show_x
        dirs.append(f"L{prev_show_x},{bottom_left.imag}")
        dirs.append("Z")
        out.append(f"""<path fill="{fill_x}" stroke="none" opacity="0.3" d="{' '.join(dirs)}" />""")
        dirs.pop()
        dirs.pop()
        dirs.pop(0)
        dirs.pop(0)
        dirs.insert(0, f"M{x2show(0)},{y2show(ys[0])}")
        out.append(
            f"""<path fill="none" stroke="{stroke_x}" stroke-width="3" d="{' '.join(dirs)}" />"""
        )

    out.append(
        f"""<rect
        x="{top_left.real}"
        y="{top_left.imag}"
        width="{span.real}"
        height="{span.imag}"
        stroke="black"
        fill="none"
        stroke-width="0.5"
    />"""
    )


if __name__ == "__main__":
    main()
