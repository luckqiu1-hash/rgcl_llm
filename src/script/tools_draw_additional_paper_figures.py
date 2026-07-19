from pathlib import Path
import textwrap

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(r"D:\pycharm\rgcl_llm")
OUT = ROOT / "src" / "paper_figures"
IMG_DIR = ROOT / "src" / "data" / "image" / "Toxicn_meme"


def ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_text(draw, xy, text, size=18, fill=(35, 35, 35), bold=False, anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor=anchor)


def wrap_text(text, width=30):
    parts = []
    for para in str(text).replace("\r", "").split("\n"):
        if not para.strip():
            continue
        parts.extend(textwrap.wrap(para, width=width, break_long_words=True, replace_whitespace=False))
    return parts or [""]


def multiline(draw, xy, text, size=16, fill=(45, 45, 45), width=30, line_gap=6, bold=False):
    x, y = xy
    fnt = font(size, bold)
    for line in wrap_text(text, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += size + line_gap
    return y


def rounded_rect(draw, box, radius=10, fill="white", outline=(180, 180, 180), width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, start, end, fill=(55, 55, 55), width=3):
    draw.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    head = 12
    for delta in (2.55, -2.55):
        x = x2 - head * math.cos(ang + delta)
        y = y2 - head * math.sin(ang + delta)
        draw.line([(x2, y2), (x, y)], fill=fill, width=width)


def fit_image(path, size):
    im = Image.open(path).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - im.width) // 2
    y = (size[1] - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def save_300(img, path):
    img.save(path, dpi=(300, 300))
    print(path)


def draw_fusion_mechanism():
    W, H = 1700, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ink = (28, 28, 28)
    muted = (92, 92, 92)
    blue = (0, 114, 178)
    orange = (213, 94, 0)
    green = (0, 158, 115)
    pale_blue = (225, 238, 248)
    pale_orange = (250, 231, 216)
    pale_green = (225, 244, 237)
    line = (135, 135, 135)

    draw_text(d, (54, 48), "Semantic explanation fusion mechanism", 30, ink, True)
    draw_text(d, (54, 88), "Semantic explanations are injected as auxiliary evidence rather than appended as raw text.", 18, muted)

    boxes = {
        "vl": (80, 210, 355, 360),
        "exp": (80, 430, 355, 580),
        "gate": (505, 160, 760, 285),
        "film": (505, 335, 760, 460),
        "res": (505, 510, 760, 635),
        "fuse": (910, 255, 1190, 485),
        "train": (1330, 178, 1605, 318),
        "clf": (1330, 438, 1605, 578),
    }
    rounded_rect(d, boxes["vl"], 10, pale_blue, blue, 2)
    rounded_rect(d, boxes["exp"], 10, pale_orange, orange, 2)
    rounded_rect(d, boxes["gate"], 10, "white", line, 2)
    rounded_rect(d, boxes["film"], 10, "white", line, 2)
    rounded_rect(d, boxes["res"], 10, "white", line, 2)
    rounded_rect(d, boxes["fuse"], 10, pale_green, green, 2)
    rounded_rect(d, boxes["train"], 10, "white", line, 2)
    rounded_rect(d, boxes["clf"], 10, "white", line, 2)

    draw_text(d, (105, 244), "Image-text\nrepresentation", 22, blue, True)
    multiline(d, (105, 300), "h_vl captures visual and OCR cues.", 15, muted, 25)
    draw_text(d, (105, 464), "Semantic explanation\nrepresentation", 22, orange, True)
    multiline(d, (105, 522), "h_exp encodes neutral meaning.", 15, muted, 25)

    draw_text(d, (530, 194), "Context-aware gate", 20, ink, True)
    multiline(d, (530, 228), "g = sigmoid(W[h_vl; h_exp])", 16, muted, 25)
    draw_text(d, (530, 370), "FiLM modulation", 20, ink, True)
    multiline(d, (530, 404), "gamma, beta = f(h_exp)\nh' = gamma * h_vl + beta", 16, muted, 24)
    draw_text(d, (530, 545), "Semantic residual", 20, ink, True)
    multiline(d, (530, 579), "h_sem = h_vl + alpha * P(h_exp)", 16, muted, 24)

    draw_text(d, (940, 302), "Controlled fusion", 22, green, True)
    multiline(d, (940, 350), "Combine gated, modulated, and residual signals to form a semantic-aware meme embedding.", 17, (45, 45, 45), 28)
    draw_text(d, (1360, 214), "Training signal", 20, ink, True)
    multiline(d, (1360, 250), "RGCL loss\n+ counterfactual evidence loss", 15, muted, 25)
    draw_text(d, (1360, 478), "Prediction", 20, ink, True)
    multiline(d, (1360, 514), "harmful / non-harmful\nclassification", 16, muted, 23)

    arrow(d, (355, 285), (505, 222), blue, 3)
    arrow(d, (355, 505), (505, 222), orange, 3)
    arrow(d, (355, 285), (505, 398), blue, 3)
    arrow(d, (355, 505), (505, 398), orange, 3)
    arrow(d, (355, 285), (505, 572), blue, 3)
    arrow(d, (355, 505), (505, 572), orange, 3)
    arrow(d, (760, 222), (910, 315), line, 3)
    arrow(d, (760, 398), (910, 370), line, 3)
    arrow(d, (760, 572), (910, 430), line, 3)
    arrow(d, (1190, 350), (1330, 248), green, 3)
    arrow(d, (1190, 390), (1330, 508), green, 3)

    draw_text(d, (80, 678), "Key point: the explanation sentence supplies explicit semantic evidence while the gate/FiLM/residual paths control how much it changes the original image-text embedding.", 18, ink)
    save_300(img, OUT / "semantic_explanation_fusion_mechanism.png")


def draw_ablation_auxiliary():
    panel_a = pd.DataFrame(
        [
            ("Baseline", 0.8117, 0.8351),
            ("+ Exp", 0.8275, 0.8526),
            ("+ CF", 0.8175, 0.8416),
            ("SEC-RGCL", 0.8392, 0.8622),
        ],
        columns=["model", "Acc", "AUC"],
    )
    panel_b = pd.DataFrame(
        [
            ("None", 0.8117, 0.8351),
            ("OCR text", 0.8217, 0.8445),
            ("Semantic exp.", 0.8275, 0.8526),
        ],
        columns=["input", "Acc", "AUC"],
    )

    def single_grouped_bar(df, title, subtitle, label_col, xlabel, output_name):
        W, H = 1180, 780
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        ink = (28, 28, 28)
        muted = (88, 88, 88)
        grid = (228, 228, 228)
        blue = (0, 114, 178)
        orange = (213, 94, 0)
        axis = (80, 80, 80)

        draw_text(d, (58, 54), title, 28, ink, True)
        draw_text(d, (58, 92), subtitle, 17, muted)
        d.rectangle((885, 64, 908, 87), fill=blue)
        draw_text(d, (920, 84), "Acc", 16, ink)
        d.rectangle((980, 64, 1003, 87), fill=orange)
        draw_text(d, (1015, 84), "AUC", 16, ink)

        x0, y0, w, h = 130, 155, 875, 430
        y_min, y_max = 0.80, 0.885

        def ymap(v):
            return y0 + h - (v - y_min) / (y_max - y_min) * h

        d.line((x0, y0, x0, y0 + h), fill=axis, width=2)
        d.line((x0, y0 + h, x0 + w, y0 + h), fill=axis, width=2)
        for tick in [0.80, 0.82, 0.84, 0.86]:
            y = ymap(tick)
            d.line((x0, y, x0 + w, y), fill=grid, width=1)
            draw_text(d, (x0 - 12, y + 5), f"{tick:.2f}", 14, muted, anchor="ra")

        n = len(df)
        group_w = w / n
        bar_w = min(54, group_w * 0.22)
        gap = 12
        for i, row in df.iterrows():
            cx = x0 + group_w * (i + 0.5)
            for metric, color, side in [("Acc", blue, -1), ("AUC", orange, 1)]:
                value = float(row[metric])
                bx = cx + side * (bar_w / 2 + gap / 2)
                y = ymap(value)
                d.rectangle((bx - bar_w / 2, y, bx + bar_w / 2, y0 + h), fill=color)
                draw_text(d, (bx, y - 13), f"{value:.3f}", 11, ink, True, anchor="ma")
            draw_text(d, (cx, y0 + h + 34), str(row[label_col]), 14, ink, anchor="ma")
            d.line((cx, y0 + h, cx, y0 + h + 6), fill=axis, width=1)

        # Light baseline emphasis for readability in print.
        d.line((x0, ymap(0.80), x0 + w, ymap(0.80)), fill=(120, 120, 120), width=1)

        draw_text(d, (x0 + w / 2, y0 + h + 58), xlabel, 15, ink, anchor="ma")
        draw_text(d, (x0 - 60, y0 + h / 2), "Score", 15, ink, anchor="ma")

        if label_col == "model":
            note = "SEC-RGCL obtains the best Acc and AUC, indicating complementary gains from semantic explanation and CF evidence learning."
        else:
            note = "Semantic explanations outperform direct OCR reuse, suggesting that explicit meaning is more useful than repeated surface text."
        multiline(d, (120, 690), note, 16, ink, 92, 6)
        save_300(img, OUT / output_name)

    single_grouped_bar(
        panel_a,
        "Module ablation results",
        "Performance comparison under different component settings.",
        "model",
        "Model setting",
        "module_ablation_results.png",
    )
    single_grouped_bar(
        panel_b,
        "Auxiliary text comparison",
        "Comparison between no auxiliary input, OCR reuse, and semantic explanation.",
        "input",
        "Auxiliary input",
        "auxiliary_text_comparison.png",
    )


def draw_case_analysis():
    df = pd.read_csv(ROOT / "src" / "case_study_candidates.csv")
    picked = [
        (
            10047,
            "Implicit abbreviation",
            "Literal: traditional virtue + CTMD. Implicit: abbreviation functions as a slang insult.",
            "Explanation maps an apparently harmless abbreviation to its pragmatic meaning.",
        ),
        (
            10512,
            "Homophonic insult",
            "Literal: 'U giver gun' + '你给我滚'. Implicit: mixed-language pun reinforces a directive insult.",
            "Explanation connects visual meme style, English-looking text, and Chinese OCR.",
        ),
        (
            5309,
            "Boundary case",
            "Literal: a neutral self-help sentence about monetizing one's skills.",
            "Explanation reveals why this is an annotation-boundary case rather than a clear semantic attack.",
        ),
    ]
    rows = []
    for sid, category, semantic, role in picked:
        r = df[df["id"] == sid].iloc[0].to_dict()
        r["category"] = category
        r["semantic"] = semantic
        r["role"] = role
        rows.append(r)

    W, H = 1750, 1130
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    ink = (28, 28, 28)
    muted = (92, 92, 92)
    blue = (0, 114, 178)
    orange = (213, 94, 0)
    green = (0, 140, 95)
    red = (190, 60, 55)
    line = (185, 185, 185)
    fill_head = (246, 246, 246)

    draw_text(d, (50, 45), "Semantic explanation case analysis", 30, ink, True)
    draw_text(d, (50, 86), "Examples show how semantic explanations expose implicit cues beyond raw OCR.", 18, muted)

    x_cols = [50, 390, 625, 900, 1320, 1695]
    headers = ["Meme", "OCR", "Semantic cue", "Model output", "Explanation role"]
    y0 = 135
    row_h = 290
    d.rectangle((50, y0, 1695, y0 + 48), fill=fill_head, outline=line)
    for i, h in enumerate(headers):
        draw_text(d, (x_cols[i] + 14, y0 + 32), h, 16, ink, True)
        d.line((x_cols[i], y0, x_cols[i], y0 + 48 + row_h * 3), fill=line, width=1)
    d.line((1695, y0, 1695, y0 + 48 + row_h * 3), fill=line, width=1)

    for ridx, r in enumerate(rows):
        top = y0 + 48 + ridx * row_h
        bottom = top + row_h
        d.line((50, top, 1695, top), fill=line, width=1)
        d.line((50, bottom, 1695, bottom), fill=line, width=1)

        meme_path = IMG_DIR / str(r["path"])
        thumb = fit_image(meme_path, (300, 225))
        ImageOps.expand(thumb, border=1, fill=(210, 210, 210))
        img.paste(thumb, (70, top + 28))
        draw_text(d, (70, top + 268), f"id={int(r['id'])} | {r['category']}", 14, muted)

        multiline(d, (410, top + 42), str(r["text"]), 18, ink, 11, 8, True)
        true = "Toxic" if int(r["label"]) == 1 else "Non-toxic"
        pred = "Toxic" if int(r["pred"]) == 1 else "Non-toxic"
        color = green if true == pred else red
        multiline(d, (920, top + 42), f"Ground truth: {true}\nPrediction: {pred}\nToxic prob.: {float(r['prob']):.3f}", 17, color, 28, 8, True)

        multiline(d, (645, top + 36), r["semantic"], 16, ink, 25, 6)
        multiline(d, (1340, top + 36), r["role"], 16, ink, 31, 6)

    d.line((50, y0 + 48 + row_h * 3, 1695, y0 + 48 + row_h * 3), fill=line, width=1)
    draw_text(d, (50, 1088), "Note: examples are selected from the local case-study candidate file; probabilities are toxic-class confidence values.", 16, muted)
    save_300(img, OUT / "semantic_case_analysis_table.png")


def main():
    ensure_out()
    draw_fusion_mechanism()
    draw_ablation_auxiliary()
    draw_case_analysis()


if __name__ == "__main__":
    main()
