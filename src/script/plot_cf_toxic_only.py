import argparse
import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Plot toxic-only counterfactual evidence analysis.")
    parser.add_argument("--sample_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_per_sample.csv")
    parser.add_argument("--summary_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_summary.csv")
    parser.add_argument("--output_png", type=str, default="./src/cf_evidence_behavior/cf_toxic_only_analysis.png")
    return parser.parse_args()


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def read_toxic_samples(path):
    by_model = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["label"]) != 1:
                continue
            by_model[row["model"]].append(
                {
                    "drop": float(row["delta_orig_minus_cf"]),
                    "orig_pred": int(row["orig_pred"]),
                    "cf_pred": int(row["cf_pred"]),
                }
            )
    return by_model


def read_summary(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def compute_stats(by_model, summary):
    stats = {}
    for model, rows in by_model.items():
        drops = [r["drop"] for r in rows]
        stats[model] = {
            "n": len(drops),
            "mean": sum(drops) / len(drops),
            "p10": percentile(drops, 0.10),
            "p25": percentile(drops, 0.25),
            "median": percentile(drops, 0.50),
            "p75": percentile(drops, 0.75),
            "p90": percentile(drops, 0.90),
            "drop_rate": sum(d > 0 for d in drops) / len(drops),
            "flip_rate": float(summary[model]["toxic_to_nontoxic_flip_rate"]) if model in summary else 0.0,
        }
    return stats


def draw_center(draw, xy, text, ft, fill=(34, 34, 34)):
    draw.text(xy, text, font=ft, fill=fill, anchor="mm")


def draw_label(draw, xy, text, ft, fill=(34, 34, 34), anchor="la"):
    draw.text(xy, text, font=ft, fill=fill, anchor=anchor)


def rounded(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_plot(stats, output_png):
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    W, H = 1680, 980
    bg = (248, 250, 252)
    ink = (24, 31, 42)
    muted = (91, 102, 120)
    grid = (222, 228, 236)
    blue = (39, 103, 190)
    blue_light = (166, 203, 244)
    red = (221, 91, 71)
    red_light = (247, 184, 172)
    green = (35, 148, 112)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    title = font(42, True)
    subtitle = font(22)
    h2 = font(26, True)
    body = font(20)
    small = font(17)
    tiny = font(15)
    num = font(24, True)

    draw_center(draw, (W // 2, 56), "Toxic Evidence Sensitivity under Counterfactual Masking", title, ink)
    draw_center(
        draw,
        (W // 2, 96),
        "Only toxic samples are analyzed. Larger probability drops indicate stronger reliance on selected toxic evidence dimensions.",
        subtitle,
        muted,
    )

    # Cards
    left_card = (70, 140, 1065, 845)
    right_card = (1100, 140, 1610, 845)
    rounded(draw, left_card, 18, "white", (230, 235, 242), 1)
    rounded(draw, right_card, 18, "white", (230, 235, 242), 1)

    draw_label(draw, (105, 182), "Distribution of toxic confidence drop", h2, ink)
    draw_label(draw, (1135, 182), "Aggregate indicators", h2, ink)

    models = [m for m in ["full", "no_cf"] if m in stats] + [m for m in stats if m not in {"full", "no_cf"}]
    colors = {"full": blue, "no_cf": red}
    light = {"full": blue_light, "no_cf": red_light}

    # Distribution plot scale.
    plot = (135, 245, 1010, 700)
    x0, y0, x1, y1 = plot
    min_v = min(min(stats[m]["p10"], stats[m]["p25"]) for m in models)
    max_v = max(max(stats[m]["p90"], stats[m]["p75"], stats[m]["mean"]) for m in models)
    min_v = min(-0.16, min_v - 0.03)
    max_v = max(0.32, max_v + 0.03)

    def xmap(v):
        return x0 + (v - min_v) / (max_v - min_v) * (x1 - x0)

    # Grid and axis.
    for tick in [-0.15, -0.05, 0.0, 0.10, 0.20, 0.30]:
        x = xmap(tick)
        draw.line((x, y0, x, y1), fill=grid, width=1)
        draw_label(draw, (x, y1 + 28), f"{tick:.2f}", tiny, muted, "mm")
    zero_x = xmap(0.0)
    draw.line((zero_x, y0 - 12, zero_x, y1 + 8), fill=(120, 130, 145), width=2)
    draw_label(draw, (x0, y1 + 64), "drop = p(toxic | original) - p(toxic | counterfactual)", small, muted)

    row_y = {"full": 365, "no_cf": 560}
    for model in models:
        s = stats[model]
        y = row_y.get(model, 365 + 160 * models.index(model))
        c = colors.get(model, green)
        lc = light.get(model, (190, 220, 205))

        draw_label(draw, (105, y), model, body, ink, "lm")

        # p10-p90 whisker
        draw.line((xmap(s["p10"]), y, xmap(s["p90"]), y), fill=lc, width=12)
        # p25-p75 interval
        rounded(draw, (xmap(s["p25"]), y - 30, xmap(s["p75"]), y + 30), 12, lc)
        # median
        draw.line((xmap(s["median"]), y - 38, xmap(s["median"]), y + 38), fill=c, width=5)
        # mean dot
        mx = xmap(s["mean"])
        draw.ellipse((mx - 13, y - 13, mx + 13, y + 13), fill=c, outline="white", width=3)
        draw_label(draw, (mx, y - 54), f"mean {s['mean']:.3f}", small, c, "mm")
        draw_label(draw, (xmap(s["median"]), y + 58), f"median {s['median']:.3f}", tiny, muted, "mm")

    # Legend.
    legend_y = 760
    draw.line((140, legend_y, 200, legend_y), fill=blue_light, width=12)
    draw.line((140, legend_y, 200, legend_y), fill=blue, width=3)
    draw_label(draw, (215, legend_y), "p10-p90 range, p25-p75 band, median line, mean dot", small, muted, "lm")

    # Right card bars.
    bar_metrics = [
        ("drop_rate", "Drop rate", "higher is better", 1.0),
        ("flip_rate", "Toxic -> non-toxic flip", "sensitivity after evidence removal", 0.35),
    ]
    bx0, bx1 = 1180, 1530
    base_y = 400
    for mi, (key, label, note, max_axis) in enumerate(bar_metrics):
        y = base_y + mi * 250
        draw_label(draw, (1135, y - 118), label, body, ink)
        draw_label(draw, (1135, y - 88), note, tiny, muted)
        draw.line((bx0, y, bx1, y), fill=grid, width=8)
        for model in models:
            s = stats[model]
            c = colors.get(model, green)
            offset = -28 if model == "full" else 28
            val = s[key]
            x_end = bx0 + min(val / max_axis, 1.0) * (bx1 - bx0)
            rounded(draw, (bx0, y + offset - 13, x_end, y + offset + 13), 7, c)
            draw_label(draw, (x_end + 12, y + offset), f"{val:.3f}", small, c, "lm")
            draw_label(draw, (1135, y + offset), model, small, muted, "lm")

    # Key takeaway banner.
    banner = (70, 875, 1610, 940)
    rounded(draw, banner, 18, (235, 243, 255), (210, 226, 250), 1)
    if "full" in stats and "no_cf" in stats:
        ratio = stats["full"]["mean"] / stats["no_cf"]["mean"] if stats["no_cf"]["mean"] != 0 else 0
        takeaway = (
            f"Key takeaway: full has a {ratio:.1f}x larger mean toxic-confidence drop "
            f"({stats['full']['mean']:.3f} vs {stats['no_cf']['mean']:.3f}) and a higher drop rate "
            f"({stats['full']['drop_rate']:.3f} vs {stats['no_cf']['drop_rate']:.3f})."
        )
        draw_center(draw, (W // 2, 907), takeaway, body, ink)

    img.save(output_png)


def print_table(stats):
    print("Toxic-only counterfactual evidence analysis")
    print("model,n,mean_drop,p25,median,p75,drop_rate,toxic_to_nontoxic_flip")
    for model, s in stats.items():
        print(
            f"{model},{s['n']},{s['mean']:.6f},{s['p25']:.6f},{s['median']:.6f},"
            f"{s['p75']:.6f},{s['drop_rate']:.6f},{s['flip_rate']:.6f}"
        )


def main():
    args = parse_args()
    samples = read_toxic_samples(args.sample_csv)
    summary = read_summary(args.summary_csv)
    stats = compute_stats(samples, summary)
    make_plot(stats, args.output_png)
    print_table(stats)
    print(f"Saved toxic-only plot to: {os.path.abspath(args.output_png)}")


if __name__ == "__main__":
    main()
