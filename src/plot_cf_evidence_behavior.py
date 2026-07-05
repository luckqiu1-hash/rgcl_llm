import argparse
import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CF evidence behavior results.")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="./src/cf_evidence_behavior/cf_evidence_summary.csv",
    )
    parser.add_argument(
        "--sample_csv",
        type=str,
        default="./src/cf_evidence_behavior/cf_evidence_per_sample.csv",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="./src/cf_evidence_behavior/cf_evidence_behavior.png",
    )
    return parser.parse_args()


def read_summary(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_samples(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            row["label"] = int(row["label"])
            row["orig_prob"] = float(row["orig_prob"])
            row["cf_prob"] = float(row["cf_prob"])
            row["delta_orig_minus_cf"] = float(row["delta_orig_minus_cf"])
            row["orig_pred"] = int(row["orig_pred"])
            row["cf_pred"] = int(row["cf_pred"])
            rows.append(row)
        return rows


def to_float(row, key):
    return float(row[key])


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def sample_distribution(samples):
    by_model = defaultdict(list)
    for row in samples:
        by_model[row["model"]].append(row)

    stats = {}
    for model, rows in by_model.items():
        toxic_drop = [r["delta_orig_minus_cf"] for r in rows if r["label"] == 1]
        nontoxic_increase = [r["cf_prob"] - r["orig_prob"] for r in rows if r["label"] == 0]
        stats[model] = {
            "toxic_drop_p25": percentile(toxic_drop, 0.25),
            "toxic_drop_p50": percentile(toxic_drop, 0.50),
            "toxic_drop_p75": percentile(toxic_drop, 0.75),
            "nontoxic_inc_p25": percentile(nontoxic_increase, 0.25),
            "nontoxic_inc_p50": percentile(nontoxic_increase, 0.50),
            "nontoxic_inc_p75": percentile(nontoxic_increase, 0.75),
        }
    return stats


def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text(draw, xy, text, font, fill=(35, 35, 35), anchor=None):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def plot(summary_rows, output_png):
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    metrics = [
        ("toxic_prob_drop_mean", "Toxic prob drop", "higher"),
        ("nontoxic_prob_increase_mean", "Non-toxic prob increase", "lower"),
        ("nontoxic_to_toxic_flip_rate", "Non-toxic -> toxic flip", "lower"),
        ("prediction_consistency", "Prediction consistency", "higher"),
    ]

    model_names = [row["model"] for row in summary_rows]
    colors = {
        "full": (55, 126, 184),
        "no_cf": (228, 107, 84),
    }
    fallback_colors = [(55, 126, 184), (228, 107, 84), (80, 160, 110), (150, 110, 180)]

    width, height = 1420, 860
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    title_font = load_font(34, bold=True)
    label_font = load_font(22, bold=True)
    tick_font = load_font(18)
    small_font = load_font(16)
    value_font = load_font(17, bold=True)

    draw_text(draw, (width // 2, 38), "Counterfactual Evidence Perturbation Analysis", title_font, anchor="ma")
    draw_text(
        draw,
        (width // 2, 78),
        "Top-k evidence dimensions are weakened; lower non-toxic increase/flip and higher toxic drop indicate better CF behavior.",
        tick_font,
        fill=(80, 80, 80),
        anchor="ma",
    )

    left, right, top, bottom = 110, 60, 145, 160
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(metrics)
    bar_w = 58

    for gi, (metric_key, label, direction) in enumerate(metrics):
        values = [to_float(row, metric_key) for row in summary_rows]
        metric_min = min(0.0, min(values))
        metric_max = max(values)
        if metric_max == metric_min:
            metric_max = metric_min + 1.0
        pad = (metric_max - metric_min) * 0.18
        metric_min -= pad
        metric_max += pad

        gx0 = left + gi * group_w
        axis_x = gx0 + group_w * 0.14
        axis_y0 = top + 30
        axis_y1 = top + plot_h - 50

        draw.line((axis_x, axis_y0, axis_x, axis_y1), fill=(210, 210, 210), width=2)
        zero_y = axis_y1 - (0 - metric_min) / (metric_max - metric_min) * (axis_y1 - axis_y0)
        draw.line((axis_x - 4, zero_y, gx0 + group_w - 26, zero_y), fill=(220, 220, 220), width=1)

        for mi, row in enumerate(summary_rows):
            value = to_float(row, metric_key)
            x = gx0 + group_w * 0.34 + mi * (bar_w + 28)
            y = axis_y1 - (value - metric_min) / (metric_max - metric_min) * (axis_y1 - axis_y0)
            base_y = zero_y
            color = colors.get(row["model"], fallback_colors[mi % len(fallback_colors)])
            y_top = min(y, base_y)
            y_bottom = max(y, base_y)
            draw.rounded_rectangle((x, y_top, x + bar_w, y_bottom), radius=5, fill=color)

            value_text = f"{value:.3f}"
            draw_text(draw, (x + bar_w / 2, y_top - 8), value_text, value_font, fill=color, anchor="mb")
            draw_text(draw, (x + bar_w / 2, axis_y1 + 12), row["model"], small_font, fill=(70, 70, 70), anchor="ma")

        arrow = "↑" if direction == "higher" else "↓"
        draw_text(draw, (gx0 + group_w * 0.50, height - 92), f"{label} {arrow}", label_font, anchor="ma")

    legend_x, legend_y = left, height - 48
    for i, row in enumerate(summary_rows):
        color = colors.get(row["model"], fallback_colors[i % len(fallback_colors)])
        x = legend_x + i * 180
        draw.rounded_rectangle((x, legend_y, x + 22, legend_y + 22), radius=4, fill=color)
        draw_text(draw, (x + 32, legend_y + 1), row["model"], tick_font)

    img.save(output_png)


def print_analysis(summary_rows, sample_stats):
    by_model = {row["model"]: row for row in summary_rows}
    if "full" in by_model and "no_cf" in by_model:
        full = by_model["full"]
        no_cf = by_model["no_cf"]
        print("Key comparisons:")
        print(f"  Original accuracy: full {to_float(full, 'orig_acc'):.4f}, no_cf {to_float(no_cf, 'orig_acc'):.4f}")
        print(
            "  Toxic prob drop: full {:.4f}, no_cf {:.4f}, gain {:.4f}".format(
                to_float(full, "toxic_prob_drop_mean"),
                to_float(no_cf, "toxic_prob_drop_mean"),
                to_float(full, "toxic_prob_drop_mean") - to_float(no_cf, "toxic_prob_drop_mean"),
            )
        )
        print(
            "  Non-toxic prob increase: full {:.4f}, no_cf {:.4f}, reduction {:.4f}".format(
                to_float(full, "nontoxic_prob_increase_mean"),
                to_float(no_cf, "nontoxic_prob_increase_mean"),
                to_float(no_cf, "nontoxic_prob_increase_mean") - to_float(full, "nontoxic_prob_increase_mean"),
            )
        )
        print(
            "  Non-toxic -> toxic flip: full {:.4f}, no_cf {:.4f}, reduction {:.4f}".format(
                to_float(full, "nontoxic_to_toxic_flip_rate"),
                to_float(no_cf, "nontoxic_to_toxic_flip_rate"),
                to_float(no_cf, "nontoxic_to_toxic_flip_rate") - to_float(full, "nontoxic_to_toxic_flip_rate"),
            )
        )

    print("\nDistribution quartiles:")
    for model, stats in sample_stats.items():
        print(
            "  {} toxic_drop p25/p50/p75: {:.4f}/{:.4f}/{:.4f}; non_toxic_inc p25/p50/p75: {:.4f}/{:.4f}/{:.4f}".format(
                model,
                stats["toxic_drop_p25"],
                stats["toxic_drop_p50"],
                stats["toxic_drop_p75"],
                stats["nontoxic_inc_p25"],
                stats["nontoxic_inc_p50"],
                stats["nontoxic_inc_p75"],
            )
        )


def main():
    args = parse_args()
    summary_rows = read_summary(args.summary_csv)
    samples = read_samples(args.sample_csv)
    sample_stats = sample_distribution(samples)
    plot(summary_rows, args.output_png)
    print_analysis(summary_rows, sample_stats)
    print(f"Saved plot to: {os.path.abspath(args.output_png)}")


if __name__ == "__main__":
    main()
