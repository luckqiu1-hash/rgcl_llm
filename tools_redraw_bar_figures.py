from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"D:/pycharm/rgcl_llm/src/paper_figures")


def font(size, bold=False):
    candidates = [
        r"C:/Windows/Fonts/arialbd.ttf" if bold else r"C:/Windows/Fonts/arial.ttf",
        r"C:/Windows/Fonts/msyhbd.ttc" if bold else r"C:/Windows/Fonts/msyh.ttc",
        r"C:/Windows/Fonts/simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered_text(draw, xy, text, fnt, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text((x - w / 2, y - h / 2), text, font=fnt, fill=fill)


def right_text(draw, xy, text, fnt, fill):
    x, y = xy
    box = draw.textbbox((0, 0), text, font=fnt)
    w = box[2] - box[0]
    h = box[3] - box[1]
    draw.text((x - w, y - h / 2), text, font=fnt, fill=fill)


def draw_grouped_bar(labels, acc, auc, title, subtitle, xlabel, filename):
    W, H = 1500, 950
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    ink = (30, 30, 30)
    muted = (84, 84, 84)
    grid = (222, 222, 222)
    axis = (58, 58, 58)
    blue = (0, 114, 178)
    orange = (213, 94, 0)

    f_title = font(40, True)
    f_sub = font(24)
    f_axis = font(24)
    f_tick = font(22)
    f_label = font(24, True)
    f_legend = font(23)

    d.text((70, 62), title, font=f_title, fill=ink)
    d.text((70, 116), subtitle, font=f_sub, fill=muted)

    # Legend with each label placed directly under its own color swatch.
    legend_y = 70
    sw = 36
    lx1 = W - 280
    lx2 = W - 155
    d.rectangle((lx1, legend_y, lx1 + sw, legend_y + sw), fill=blue)
    d.rectangle((lx2, legend_y, lx2 + sw, legend_y + sw), fill=orange)
    centered_text(d, (lx1 + sw / 2, legend_y + sw + 22), "Acc", f_legend, ink)
    centered_text(d, (lx2 + sw / 2, legend_y + sw + 22), "AUC", f_legend, ink)

    x0, y0 = 165, 205
    plot_w, plot_h = 1140, 520
    y_min, y_max = 0.800, 0.872

    def ymap(v):
        return y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Axes and grid.
    d.line((x0, y0, x0, y0 + plot_h), fill=axis, width=3)
    d.line((x0, y0 + plot_h, x0 + plot_w, y0 + plot_h), fill=axis, width=3)
    for tick in [0.80, 0.82, 0.84, 0.86]:
        y = ymap(tick)
        d.line((x0, y, x0 + plot_w, y), fill=grid, width=1)
        right_text(d, (x0 - 18, y), f"{tick:.2f}", f_tick, muted)

    n = len(labels)
    group_w = plot_w / n
    bar_w = min(72, group_w * 0.24)
    gap = 18
    baseline = y0 + plot_h

    for i, label in enumerate(labels):
        cx = x0 + group_w * (i + 0.5)
        positions = [
            (cx - (bar_w + gap) / 2, acc[i], blue),
            (cx + (bar_w + gap) / 2, auc[i], orange),
        ]
        for bx, value, color in positions:
            y = ymap(value)
            d.rectangle((bx - bar_w / 2, y, bx + bar_w / 2, baseline), fill=color)
            # Larger, centered value labels, each tied to its own bar center.
            centered_text(d, (bx, y - 18), f"{value:.3f}", f_label, ink)
        centered_text(d, (cx, baseline + 53), label, f_tick, ink)
        d.line((cx, baseline, cx, baseline + 10), fill=axis, width=2)

    centered_text(d, (x0 + plot_w / 2, baseline + 93), xlabel, f_axis, ink)
    centered_text(d, (82, y0 + plot_h / 2), "Score", f_axis, ink)

    img.save(OUT / filename, dpi=(300, 300))
    print(OUT / filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    draw_grouped_bar(
        ["Baseline", "+ Exp", "+ CF", "SEC-RGCL"],
        [0.8117, 0.8275, 0.8175, 0.8392],
        [0.8351, 0.8526, 0.8416, 0.8622],
        "Module ablation results",
        "Acc and AUC under different component settings.",
        "Model setting",
        "module_ablation_results.png",
    )
    draw_grouped_bar(
        ["None", "OCR text", "Semantic exp."],
        [0.8117, 0.8217, 0.8275],
        [0.8351, 0.8445, 0.8526],
        "Auxiliary text comparison",
        "Acc and AUC under different auxiliary text inputs.",
        "Auxiliary input",
        "auxiliary_text_comparison.png",
    )


if __name__ == "__main__":
    main()
