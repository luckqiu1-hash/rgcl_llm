import argparse
import csv
import os
from collections import defaultdict
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Create publication-ready toxic-only CF evidence figure.")
    parser.add_argument("--sample_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_per_sample.csv")
    parser.add_argument("--summary_csv", type=str, default="./src/cf_evidence_behavior/cf_evidence_summary.csv")
    parser.add_argument("--output_svg", type=str, default="./src/cf_evidence_behavior/cf_toxic_only_paper.svg")
    parser.add_argument("--output_png", type=str, default="./src/cf_evidence_behavior/cf_toxic_only_paper.png")
    parser.add_argument("--show_takeaway", action="store_true")
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


def svg_text(x, y, text, size=16, weight="400", color="#1f2937", anchor="start"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{escape(text)}</text>"
    )


def svg_line(x1, y1, x2, y2, color="#d1d5db", width=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def svg_rect(x, y, w, h, fill, stroke="none", width=1.0, radius=0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
    )


def svg_circle(cx, cy, r, fill, stroke="white", width=2.0):
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"/>'
    )


def make_svg(stats, output_svg, show_takeaway=False):
    os.makedirs(os.path.dirname(os.path.abspath(output_svg)), exist_ok=True)

    models = ordered_models(stats)
    colors = {"full": "#2563eb", "no_cf": "#ef4444"}
    light = {"full": "#bfdbfe", "no_cf": "#fecaca"}
    ink = "#111827"
    muted = "#4b5563"
    grid = "#e5e7eb"

    w, h = 980, 520
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        svg_rect(0, 0, w, h, "#ffffff"),
        svg_text(42, 42, "(a) Toxic confidence drop distribution", 18, "700", ink),
        svg_text(588, 42, "(b) Aggregate toxic-only indicators", 18, "700", ink),
    ]

    # Panel A: interval distribution.
    x0, x1 = 88, 540
    y0, y1 = 92, 380
    min_v, max_v = -0.15, 0.32

    def xmap(v):
        return x0 + (v - min_v) / (max_v - min_v) * (x1 - x0)

    for tick in [-0.15, -0.05, 0.0, 0.10, 0.20, 0.30]:
        x = xmap(tick)
        parts.append(svg_line(x, y0, x, y1, grid, 1))
        parts.append(svg_text(x, y1 + 24, f"{tick:.2f}", 13, "400", muted, "middle"))
    parts.append(svg_line(xmap(0.0), y0 - 8, xmap(0.0), y1 + 6, "#6b7280", 1.6))
    parts.append(svg_line(x0, y1, x1, y1, "#9ca3af", 1.2))
    parts.append(svg_text((x0 + x1) / 2, y1 + 50, "Drop = p(toxic | original) - p(toxic | counterfactual)", 13, "400", muted, "middle"))

    row_y = {"full": 170, "no_cf": 280}
    for model in models:
        s = stats[model]
        y = row_y.get(model, 170 + 100 * models.index(model))
        c = colors.get(model, "#059669")
        lc = light.get(model, "#bbf7d0")
        parts.append(svg_text(42, y + 5, model, 15, "600", ink))
        parts.append(svg_line(xmap(s["p10"]), y, xmap(s["p90"]), y, lc, 8))
        parts.append(svg_rect(xmap(s["p25"]), y - 19, xmap(s["p75"]) - xmap(s["p25"]), 38, lc, "none", radius=8))
        parts.append(svg_line(xmap(s["median"]), y - 25, xmap(s["median"]), y + 25, c, 3))
        parts.append(svg_circle(xmap(s["mean"]), y, 7, c))
        parts.append(svg_text(xmap(s["mean"]), y - 32, f"mean {s['mean']:.3f}", 13, "700", c, "middle"))
        parts.append(svg_text(xmap(s["median"]), y + 44, f"median {s['median']:.3f}", 12, "400", muted, "middle"))

    # Panel B: bars.
    bx0, bx1 = 640, 900
    metrics = [
        ("mean", "Mean drop", 0.18),
        ("drop_rate", "Drop rate", 1.0),
        ("flip_rate", "Toxic -> non-toxic flip", 0.35),
    ]
    for i, (key, label, max_axis) in enumerate(metrics):
        y = 116 + i * 108
        parts.append(svg_text(588, y, label, 15, "600", ink))
        parts.append(svg_line(bx0, y + 28, bx1, y + 28, "#dbe1ea", 5))
        for j, model in enumerate(models):
            s = stats[model]
            val = s[key]
            c = colors.get(model, "#059669")
            yy = y + 15 + j * 28
            x_end = bx0 + min(val / max_axis, 1.0) * (bx1 - bx0)
            parts.append(svg_rect(bx0, yy, x_end - bx0, 18, c, radius=5))
            parts.append(svg_text(588, yy + 14, model, 13, "400", muted))
            parts.append(svg_text(x_end + 8, yy + 14, f"{val:.3f}", 13, "700", c))

    # Compact takeaway.
    if show_takeaway and "full" in stats and "no_cf" in stats and stats["no_cf"]["mean"] != 0:
        ratio = stats["full"]["mean"] / stats["no_cf"]["mean"]
        takeaway = (
            f"Full shows {ratio:.1f}x larger mean drop "
            f"({stats['full']['mean']:.3f} vs {stats['no_cf']['mean']:.3f}) "
            f"and higher drop rate ({stats['full']['drop_rate']:.3f} vs {stats['no_cf']['drop_rate']:.3f})."
        )
        parts.append(svg_rect(42, 450, 896, 42, "#f3f7ff", "#dbeafe", radius=8))
        parts.append(svg_text(490, 477, takeaway, 15, "600", "#1e3a8a", "middle"))

    parts.append("</svg>")
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_png(stats, output_png, show_takeaway=False):
    # High-resolution raster fallback that visually matches the SVG.
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    scale = 3
    w, h = 980 * scale, 520 * scale
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    def s(v):
        return int(round(v * scale))

    def text(x, y, value, size=16, fill=(17, 24, 39), bold=False, anchor="la"):
        d.text((s(x), s(y)), value, font=font(size * scale, bold), fill=fill, anchor=anchor)

    def line(x1, y1, x2, y2, fill=(229, 231, 235), width=1):
        d.line((s(x1), s(y1), s(x2), s(y2)), fill=fill, width=s(width))

    def rect(x, y, ww, hh, fill, radius=0, outline=None):
        d.rounded_rectangle((s(x), s(y), s(x + ww), s(y + hh)), radius=s(radius), fill=fill, outline=outline)

    def circle(cx, cy, r, fill):
        d.ellipse((s(cx - r), s(cy - r), s(cx + r), s(cy + r)), fill=fill, outline="white", width=s(2))

    models = ordered_models(stats)
    colors = {"full": (37, 99, 235), "no_cf": (239, 68, 68)}
    light = {"full": (191, 219, 254), "no_cf": (254, 202, 202)}
    ink = (17, 24, 39)
    muted = (75, 85, 99)

    text(42, 42, "(a) Toxic confidence drop distribution", 18, ink, True)
    text(588, 42, "(b) Aggregate toxic-only indicators", 18, ink, True)

    x0, x1, y0, y1 = 88, 540, 92, 380
    min_v, max_v = -0.15, 0.32

    def xmap(v):
        return x0 + (v - min_v) / (max_v - min_v) * (x1 - x0)

    for tick in [-0.15, -0.05, 0.0, 0.10, 0.20, 0.30]:
        x = xmap(tick)
        line(x, y0, x, y1)
        text(x, y1 + 24, f"{tick:.2f}", 13, muted, anchor="mm")
    line(xmap(0), y0 - 8, xmap(0), y1 + 6, (107, 114, 128), 1.6)
    line(x0, y1, x1, y1, (156, 163, 175), 1.2)
    text((x0 + x1) / 2, y1 + 50, "Drop = p(toxic | original) - p(toxic | counterfactual)", 13, muted, anchor="mm")

    row_y = {"full": 170, "no_cf": 280}
    for model in models:
        st = stats[model]
        y = row_y.get(model, 170 + 100 * models.index(model))
        c = colors.get(model, (5, 150, 105))
        lc = light.get(model, (187, 247, 208))
        text(42, y + 5, model, 15, ink, True)
        line(xmap(st["p10"]), y, xmap(st["p90"]), y, lc, 8)
        rect(xmap(st["p25"]), y - 19, xmap(st["p75"]) - xmap(st["p25"]), 38, lc, 8)
        line(xmap(st["median"]), y - 25, xmap(st["median"]), y + 25, c, 3)
        circle(xmap(st["mean"]), y, 7, c)
        text(xmap(st["mean"]), y - 32, f"mean {st['mean']:.3f}", 13, c, True, "mm")
        text(xmap(st["median"]), y + 44, f"median {st['median']:.3f}", 12, muted, anchor="mm")

    bx0, bx1 = 640, 900
    metrics = [("mean", "Mean drop", 0.18), ("drop_rate", "Drop rate", 1.0), ("flip_rate", "Toxic -> non-toxic flip", 0.35)]
    for i, (key, label, max_axis) in enumerate(metrics):
        y = 116 + i * 108
        text(588, y, label, 15, ink, True)
        line(bx0, y + 28, bx1, y + 28, (219, 225, 234), 5)
        for j, model in enumerate(models):
            st = stats[model]
            val = st[key]
            c = colors.get(model, (5, 150, 105))
            yy = y + 15 + j * 28
            x_end = bx0 + min(val / max_axis, 1.0) * (bx1 - bx0)
            rect(bx0, yy, x_end - bx0, 18, c, 5)
            text(588, yy + 14, model, 13, muted)
            text(x_end + 8, yy + 14, f"{val:.3f}", 13, c, True)

    if show_takeaway and "full" in stats and "no_cf" in stats and stats["no_cf"]["mean"] != 0:
        ratio = stats["full"]["mean"] / stats["no_cf"]["mean"]
        takeaway = f"Full shows {ratio:.1f}x larger mean drop ({stats['full']['mean']:.3f} vs {stats['no_cf']['mean']:.3f}) and higher drop rate ({stats['full']['drop_rate']:.3f} vs {stats['no_cf']['drop_rate']:.3f})."
        rect(42, 450, 896, 42, (243, 247, 255), 8, (219, 234, 254))
        text(490, 477, takeaway, 15, (30, 58, 138), True, "mm")

    img.save(output_png, dpi=(300, 300))


def print_table(stats):
    print("Publication toxic-only table")
    print("model,n,mean_drop,p10,p25,median,p75,p90,drop_rate,toxic_to_nontoxic_flip")
    for model in ordered_models(stats):
        s = stats[model]
        print(
            f"{model},{s['n']},{s['mean']:.6f},{s['p10']:.6f},{s['p25']:.6f},"
            f"{s['median']:.6f},{s['p75']:.6f},{s['p90']:.6f},"
            f"{s['drop_rate']:.6f},{s['flip_rate']:.6f}"
        )


def main():
    args = parse_args()
    samples = read_toxic_samples(args.sample_csv)
    summary = read_summary(args.summary_csv)
    stats = compute_stats(samples, summary)
    make_svg(stats, args.output_svg, args.show_takeaway)
    make_png(stats, args.output_png, args.show_takeaway)
    print_table(stats)
    print(f"Saved SVG to: {os.path.abspath(args.output_svg)}")
    print(f"Saved PNG to: {os.path.abspath(args.output_png)}")


if __name__ == "__main__":
    main()
