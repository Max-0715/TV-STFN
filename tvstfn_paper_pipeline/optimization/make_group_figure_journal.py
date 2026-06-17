import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = "/data/workplace/jwx/TV-STFN"
IN_DIR = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_section3_figures")
OUT_DIR = IN_DIR

A_PATH = os.path.join(IN_DIR, "figure_3A_overall_compare.png")
B_PATH = os.path.join(IN_DIR, "figure_3B_cliff_delta.png")
C_PATH = os.path.join(IN_DIR, "figure_3C_stratified_f1.png")


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_width(im, target_w):
    w, h = im.size
    ratio = target_w / float(w)
    new_h = int(h * ratio)
    return im.resize((target_w, new_h), Image.Resampling.LANCZOS)


def add_panel_label(im, label, margin=12):
    canvas = Image.new("RGB", (im.size[0] + margin * 2, im.size[1] + margin * 2), (255, 255, 255))
    canvas.paste(im, (margin, margin))

    draw = ImageDraw.Draw(canvas)
    font = load_font(34, bold=True)

    x0, y0 = margin + 6, margin + 6
    x1, y1 = x0 + 44, y0 + 40
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(180, 180, 180), width=1)
    draw.text((x0 + 10, y0 - 2), label, fill=(20, 20, 20), font=font)
    return canvas


def build_figure(a_path, b_path, c_path):
    for p in [a_path, b_path, c_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    c = Image.open(c_path).convert("RGB")

    # Journal-like layout: top A/B, bottom C centered
    top_w = 980
    c_w = top_w * 2 + 36

    a = ImageOps.expand(fit_width(a, top_w), border=1, fill=(220, 220, 220))
    b = ImageOps.expand(fit_width(b, top_w), border=1, fill=(220, 220, 220))
    c = ImageOps.expand(fit_width(c, c_w), border=1, fill=(220, 220, 220))

    pa = add_panel_label(a, "A")
    pb = add_panel_label(b, "B")
    pc = add_panel_label(c, "C")

    gap = 18
    width = pa.size[0] + pb.size[0] + gap
    top_h = max(pa.size[1], pb.size[1])
    height = top_h + gap + pc.size[1]

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(pa, (0, 0))
    canvas.paste(pb, (pa.size[0] + gap, 0))

    x_c = (width - pc.size[0]) // 2
    y_c = top_h + gap
    canvas.paste(pc, (x_c, y_c))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(210, 210, 210), width=1)
    return canvas


def write_texts(out_dir):
    cap_en = os.path.join(out_dir, "figure_3_ABC_caption_en.md")
    cap_zh = os.path.join(out_dir, "figure_3_ABC_caption_zh.md")
    para_en = os.path.join(out_dir, "section3_results_paragraph_en.md")
    para_zh = os.path.join(out_dir, "section3_results_paragraph_zh.md")

    with open(cap_en, "w", encoding="utf-8") as f:
        f.write(
            "Figure 3. Integrated evaluation of TV-STFN from global performance, activity-cliff sensitivity, and stratified robustness. "
            "(A) Overall classification/regression metrics comparing tv5overnight_1 and hard_v1_overnight. "
            "(B) True versus predicted permeability differences for top-10 activity-cliff pairs. "
            "(C) Stratified F1 across MW and TPSA bins, showing degradation in highly complex regions.\n"
        )

    with open(cap_zh, "w", encoding="utf-8") as f:
        f.write(
            "图3. 从整体性能、活性悬崖敏感性和分层稳健性三个维度对 TV-STFN 进行综合评估。"
            "(A) tv5overnight_1 与 hard_v1_overnight 的分类/回归总体指标对比；"
            "(B) 前10组活性悬崖样本对的真实与预测渗透率差值对比；"
            "(C) 基于 MW 与 TPSA 分箱的 F1 分层表现，显示模型在高复杂区域存在性能下降。\n"
        )

    with open(para_en, "w", encoding="utf-8") as f:
        f.write(
            "As shown in Figure 3A, the baseline configuration (tv5overnight_1) remains slightly stronger than hard_v1_overnight in overall metrics, "
            "including ACC, F1, MCC, AUROC, and RMSE. Figure 3B further indicates that both settings still produce non-negligible errors when modeling top activity-cliff pairs, "
            "suggesting limited sensitivity to abrupt local structure-activity changes. In Figure 3C, a clear drop in F1 is observed in high-TPSA and high-MW regions, "
            "confirming that physicochemical complexity remains a major bottleneck. Collectively, these findings support future optimization with cliff-aware hard-sample training and structure-focused data augmentation.\n"
        )

    with open(para_zh, "w", encoding="utf-8") as f:
        f.write(
            "如图3A所示，基线配置 tv5overnight_1 在 ACC、F1、MCC、AUROC 和 RMSE 等总体指标上仍略优于 hard_v1_overnight。"
            "图3B显示，两种设置在前10组活性悬崖样本对上的差值拟合仍存在明显误差，说明模型对局部突变型构效变化的刻画能力仍有限。"
            "图3C进一步表明，在高 TPSA 与高 MW 区域，F1 出现系统性下降，验证了理化复杂性是当前泛化性能的关键瓶颈。"
            "综合来看，后续应优先采用悬崖感知的难样本训练与结构约束增强策略。\n"
        )

    return [cap_en, cap_zh, para_en, para_zh]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    merged = build_figure(A_PATH, B_PATH, C_PATH)

    out_png = os.path.join(OUT_DIR, "figure_3_ABC_journal_style.png")
    out_tif = os.path.join(OUT_DIR, "figure_3_ABC_journal_style.tif")
    out_pdf = os.path.join(OUT_DIR, "figure_3_ABC_journal_style.pdf")

    merged.save(out_png, dpi=(300, 300))
    merged.save(out_tif, dpi=(300, 300), compression="tiff_lzw")
    merged.save(out_pdf, resolution=300.0)

    text_files = write_texts(OUT_DIR)

    print(out_png)
    print(out_tif)
    print(out_pdf)
    for p in text_files:
        print(p)


if __name__ == "__main__":
    main()
