import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:/pycharm/rgcl_llm")
DATA_DIR = ROOT / "src" / "cf_evidence_behavior"
SAMPLE_CSV = DATA_DIR / "cf_evidence_per_sample.csv"
SUMMARY_CSV = DATA_DIR / "cf_evidence_summary.csv"
OUT_DIST = DATA_DIR / "cf_drop_distribution_split.png"
OUT_IND = DATA_DIR / "cf_response_indicators_split.png"


def font(size, bold=False):
    candidates = [
        r"C:/Windows/Fonts/arialbd.ttf" if bold else r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/calibrib.ttf" if bold else r"C:/Windows/Fonts/calibri.ttf",
        r"C:/Windows/Fonts/msyhbd.ttc" if bold else r"C:/Windows/Fonts/msyh.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_center(draw, xy, text, fnt, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def text_right(draw, xy, text, fnt, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]), y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def display_model(model):
    return "with_cf" if model == "full" else model


def read_data():
    samples = defaultdict(list)
    with SAMPLE_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["label"]) == 1:
                samples[row["model"]].append(float(row["delta_orig_minus_cf"]))

    summary = {}
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            summary[row["model"]] = row

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
            "drop_rate": float(summary[model]["toxic_prob_drop_rate"]),
            "flip_rate": float(summary[model]["toxic_to_nontoxic_flip_rate"]),
        }
    return stats


def ordered(stats):
    return [m for m in ("full", "no_cf") if m in stats] + [m for m in stats if m not in ("full", "no_cf")]


def draw_distribution(stats):
    W, H = 1500, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    ink = (30, 30, 30)
    muted = (90, 90, 90)
    grid = (224, 224, 224)
    axis = (70, 70, 70)
    blue = (0, 114, 178)
    blue_light = (214, 230, 244)
    orange = (213, 94, 0)
    orange_light = (247, 220, 201)
    colors = {"full": blue, "no_cf": orange}
    lights = {"full": blue_light, "no_cf": orange_light}

    f_title = font(38, True)
    f_sub = font(23)
    f_axis = font(24)
    f_tick = font(21)
    f_label = font(23, True)
    f_small = font(20)

    d.text((70, 55), "Toxic-confidence drop distribution", font=f_title, fill=ink)
    d.text((70, 108), "Effect of weakening high-contribution evidence dimensions on toxic samples.", font=f_sub, fill=muted)

    x0, x1 = 230, 1350
    y_top, y_bottom = 200, 545
    min_v, max_v = -0.15, 0.32

    def xmap(v):
        return x0 + (v - min_v) / (max_v - min_v) * (x1 - x0)

    for tick in [-0.15, -0.05, 0.00, 0.10, 0.20, 0.30]:
        x = xmap(tick)
        d.line((x, y_top, x, y_bottom), fill=grid, width=1)
        text_center(d, (x, y_bottom + 38), f"{tick:.2f}", f_tick, muted)

    d.line((xmap(0), y_top - 10, xmap(0), y_bottom + 10), fill=axis, width=3)
    d.line((x0, y_bottom, x1, y_bottom), fill=axis, width=3)
    text_center(d, ((x0 + x1) / 2, y_bottom + 88), "p(toxic | original) - p(toxic | counterfactual)", f_axis, ink)
    d.text((910, 178), "line: p10-p90; box: p25-p75; dot: mean", font=f_small, fill=muted)

    row_y = {"full": 300, "no_cf": 435}
    for model in ordered(stats):
        st = stats[model]
        y = row_y.get(model, 300)
        c = colors.get(model, ink)
        lc = lights.get(model, (230, 230, 230))
        d.text((70, y - 18), display_model(model), font=f_label, fill=ink)
        d.line((xmap(st["p10"]), y, xmap(st["p90"]), y), fill=c, width=7)
        d.rectangle((xmap(st["p25"]), y - 28, xmap(st["p75"]), y + 28), fill=lc, outline=c, width=3)
        d.line((xmap(st["median"]), y - 36, xmap(st["median"]), y + 36), fill=c, width=5)
        d.ellipse((xmap(st["mean"]) - 10, y - 10, xmap(st["mean"]) + 10, y + 10), fill=c, outline="white", width=3)

    img.save(OUT_DIST, dpi=(300, 300))
    print(OUT_DIST)


def draw_indicators(stats):
    W, H = 1500, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    ink = (30, 30, 30)
    muted = (90, 90, 90)
    grid = (224, 224, 224)
    axis = (70, 70, 70)
    blue = (0, 114, 178)
    orange = (213, 94, 0)
    colors = {"full": blue, "no_cf": orange}

    f_title = font(38, True)
    f_sub = font(23)
    f_axis = font(22)
    f_label = font(23, True)
    f_value = font(22, True)

    d.text((70, 55), "Counterfactual-response indicators", font=f_title, fill=ink)
    d.text((70, 108), "Aggregate response after weakening high-contribution dimensions in toxic samples.", font=f_sub, fill=muted)

    # Legend.
    d.rectangle((1180, 66, 1220, 106), fill=blue)
    d.text((1235, 70), "with_cf", font=f_axis, fill=ink)
    d.rectangle((1180, 118, 1220, 158), fill=orange)
    d.text((1235, 122), "no_cf", font=f_axis, fill=ink)

    x0, x1 = 560, 1320
    metrics = [
        ("mean", "Mean drop", 0.18, [0.00, 0.06, 0.12, 0.18]),
        ("drop_rate", "Drop rate", 1.0, [0.00, 0.25, 0.50, 0.75, 1.00]),
        ("flip_rate", "Toxic -> non-toxic flip", 0.35, [0.00, 0.10, 0.20, 0.30]),
    ]

    def xmap(v, max_axis):
        return x0 + min(max(v, 0) / max_axis, 1.0) * (x1 - x0)

    for i, (key, label, max_axis, ticks) in enumerate(metrics):
        y = 250 + i * 150
        d.text((70, y - 20), label, font=f_label, fill=ink)
        d.line((x0, y, x1, y), fill=axis, width=3)
        for tick in ticks:
            x = xmap(tick, max_axis)
            d.line((x, y - 10, x, y + 10), fill=axis, width=2)
            tick_label = f"{tick:.2f}" if max_axis < 1 else f"{tick:.2f}".rstrip("0").rstrip(".")
            text_center(d, (x, y + 38), tick_label, f_axis, muted)
            if tick not in (ticks[0], ticks[-1]):
                d.line((x, y - 58, x, y + 58), fill=grid, width=1)

        values = []
        for model in ordered(stats):
            value = stats[model][key]
            values.append((model, xmap(value, max_axis), value))
        if len(values) >= 2:
            d.line((values[0][1], y, values[1][1], y), fill=(165, 165, 165), width=4)
        for model, x, value in values:
            c = colors.get(model, ink)
            d.ellipse((x - 13, y - 13, x + 13, y + 13), fill=c, outline="white", width=3)
            if model == "full":
                d.text((x + 22, y - 32), f"{value:.3f}", font=f_value, fill=c)
            else:
                text_right(d, (x - 22, y - 20), f"{value:.3f}", f_value, c)

    img.save(OUT_IND, dpi=(300, 300))
    print(OUT_IND)


def main():
    stats = read_data()
    draw_distribution(stats)
    draw_indicators(stats)


if __name__ == "__main__":
    main()
