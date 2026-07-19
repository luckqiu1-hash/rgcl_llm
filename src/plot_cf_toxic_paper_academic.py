import argparse
import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Create an academic-style CF evidence figure.")
    parser.add_argument("--sample_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_per_sample.csv")
    parser.add_argument("--summary_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_summary.csv")
    parser.add_argument("--output_png", type=str, default="./src/cf_evidence_behavior/cf_toxic_only_paper_academic.png")
    return parser.parse_args()


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
            if int(row["label"]) == 1:
                by_model[row["model"]].append(float(row["delta_orig_minus_cf"]))
    return by_model


def read_summary(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["model"]: row for row in csv.DictReader(f)}


def compute_stats(samples, summary):
    stats = {}
    for model, drops in samples.items():
        stats[model] = {
            "n": len(drops),
            "mean": sum(drops) / len(drops),
            "p10": percentile(drops, 0.10),
            "p25": percentile(drops, 0.25),
            "median": percentile(drops, 0.50),
            "p75": percentile(drops, 0.75),
            "p90": percentile(drops, 0.90),
            "drop_rate": sum(value > 0 for value in drops) / len(drops),
            "flip_rate": float(summary[model]["toxic_to_nontoxic_flip_rate"]) if model in summary else 0.0,
        }
    return stats


def ordered_models(stats):
    preferred = [model for model in ["full", "no_cf"] if model in stats]
    return preferred + [model for model in stats if model not in preferred]


def display_model(model):
    return "with_cf" if model == "full" else model


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_text(draw, xy, text, size, fill, bold=False, anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def make_png(stats, output_png):
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)

    scale = 3
    W, H = 1500, 560
    img = Image.new("RGB", (W * scale, H * scale), "white")
    d = ImageDraw.Draw(img)

    def S(x):
        return int(round(x * scale))

    def line(x1, y1, x2, y2, color, width=1):
        d.line((S(x1), S(y1), S(x2), S(y2)), fill=color, width=max(1, S(width)))

    def rect(x1, y1, x2, y2, fill, outline=None, width=1):
        d.rectangle((S(x1), S(y1), S(x2), S(y2)), fill=fill, outline=outline, width=max(1, S(width)))

    def circle(x, y, r, fill, outline="white", width=1):
        d.ellipse((S(x - r), S(y - r), S(x + r), S(y + r)), fill=fill, outline=outline, width=max(1, S(width)))

    def T(x, y, text, size=18, color=(32, 32, 32), bold=False, anchor="la"):
        draw_text(d, (S(x), S(y)), text, size * scale, color, bold, anchor)

    ink = (30, 30, 30)
    muted = (92, 92, 92)
    grid = (229, 229, 229)
    axis = (88, 88, 88)
    blue = (0, 114, 178)
    blue_light = (214, 230, 244)
    orange = (213, 94, 0)
    orange_light = (247, 220, 201)
    colors = {"full": blue, "no_cf": orange}
    lights = {"full": blue_light, "no_cf": orange_light}

    # Panel labels and concise titles.
    T(42, 44, "A", 24, ink, True)
    T(82, 44, "Toxic-confidence drop distribution", 20, ink)
    T(840, 44, "B", 24, ink, True)
    T(880, 44, "Counterfactual-response indicators", 20, ink)

    # Compact in-panel legend.
    circle(1215, 47, 6, blue, outline=blue)
    T(1230, 52, "with_cf", 14, ink)
    circle(1308, 47, 6, orange, outline=orange)
    T(1323, 52, "no_cf", 14, ink)

    # Panel A: compact horizontal box/range plot.
    x0, x1 = 140, 735
    y_top, y_bottom = 110, 390
    min_v, max_v = -0.15, 0.32

    def xmap(v):
        return x0 + (v - min_v) / (max_v - min_v) * (x1 - x0)

    for tick in [-0.15, -0.05, 0.00, 0.10, 0.20, 0.30]:
        x = xmap(tick)
        line(x, y_top, x, y_bottom, grid, 0.6)
        T(x, y_bottom + 25, f"{tick:.2f}", 13, muted, anchor="ma")
    line(xmap(0), y_top - 6, xmap(0), y_bottom + 6, axis, 1.0)
    line(x0, y_bottom, x1, y_bottom, axis, 0.9)
    T((x0 + x1) / 2, y_bottom + 62, "p(toxic | original) - p(toxic | counterfactual)", 14, ink, anchor="ma")

    row_y = {"full": 185, "no_cf": 305}
    for model in ordered_models(stats):
        st = stats[model]
        y = row_y.get(model, 200)
        c = colors.get(model, (80, 80, 80))
        lc = lights.get(model, (220, 220, 220))
        T(48, y + 5, display_model(model), 15, ink)
        # p10-p90 whisker, p25-p75 interval, median, mean.
        line(xmap(st["p10"]), y, xmap(st["p90"]), y, c, 1.8)
        rect(xmap(st["p25"]), y - 16, xmap(st["p75"]), y + 16, lc, outline=c, width=0.9)
        line(xmap(st["median"]), y - 20, xmap(st["median"]), y + 20, c, 1.9)
        circle(xmap(st["mean"]), y, 5, c, width=1)
        T(xmap(st["mean"]), y - 34, f"mean {st['mean']:.3f}", 12, c, True, anchor="ma")
        T(xmap(st["median"]), y + 38, f"median {st['median']:.3f}", 12, muted, anchor="ma")

    T(470, 105, "line: p10-p90; box: p25-p75; dot: mean", 11, muted)

    # Panel B: dot/line comparison for aggregate metrics.
    bx0, bx1 = 955, 1400
    metrics = [
        ("mean", "Mean drop", 0.18, [0.00, 0.06, 0.12, 0.18]),
        ("drop_rate", "Drop rate", 1.0, [0.0, 0.25, 0.50, 0.75, 1.0]),
        ("flip_rate", "Toxic -> non-toxic flip", 0.35, [0.0, 0.10, 0.20, 0.30]),
    ]

    def bxmap(value, max_axis):
        return bx0 + min(max(value, 0) / max_axis, 1.0) * (bx1 - bx0)

    for i, (key, label, max_axis, ticks) in enumerate(metrics):
        y = 138 + i * 130
        T(840, y - 30, label, 15, ink)
        line(bx0, y, bx1, y, axis, 0.9)
        for tick in ticks:
            x = bxmap(tick, max_axis)
            line(x, y - 5, x, y + 5, axis, 0.8)
            tick_label = f"{tick:.2f}" if max_axis < 1 else f"{tick:.2f}".rstrip("0").rstrip(".")
            T(x, y + 22, tick_label, 11, muted, anchor="ma")
        values = []
        for model in ordered_models(stats):
            st = stats[model]
            x = bxmap(st[key], max_axis)
            values.append((model, x, st[key]))
        if len(values) >= 2:
            line(values[0][1], y, values[1][1], y, (174, 174, 174), 1.0)
        for model, x, val in values:
            c = colors.get(model, (80, 80, 80))
            circle(x, y, 6, c, outline="white", width=1.2)
            if model == "full":
                T(min(x + 18, bx1 - 34), y - 17, f"{val:.3f}", 11, c, True)
            else:
                T(max(x - 18, bx0 + 8), y - 17, f"{val:.3f}", 11, c, True, anchor="ra")

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(output_png, dpi=(300, 300))


def main():
    args = parse_args()
    samples = read_toxic_samples(args.sample_csv)
    summary = read_summary(args.summary_csv)
    stats = compute_stats(samples, summary)
    make_png(stats, args.output_png)
    print(f"Saved academic PNG to: {os.path.abspath(args.output_png)}")


if __name__ == "__main__":
    main()
