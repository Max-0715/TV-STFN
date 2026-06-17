import os
from PIL import Image, ImageOps, ImageDraw, ImageFont


ROOT = "/data/workplace/jwx/TV-STFN"
IN_DIR = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_section3_figures")
OUT_DIR = os.path.join(ROOT, "tvstfn_paper_pipeline/outputs/paper_section3_figures")
os.makedirs(OUT_DIR, exist_ok=True)

A_PATH = os.path.join(IN_DIR, "figure_3A_overall_compare.png")
B_PATH = os.path.join(IN_DIR, "figure_3B_cliff_delta.png")
C_PATH = os.path.join(IN_DIR, "figure_3C_stratified_f1.png")


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def fit_width(im: Image.Image, target_w: int) -> Image.Image:
    w, h = im.size
    ratio = target_w / float(w)
    nh = int(h * ratio)
    return im.resize((target_w, nh), Image.Resampling.LANCZOS)


def panel(im: Image.Image, label: str, title: str, pad: int, border: int, width: int):
    im = fit_width(im, width)
    # keep white background style consistent with MSF figures
    im = ImageOps.expand(im, border=border, fill=(220, 220, 220))

    title_h = 48
    panel_img = Image.new("RGB", (im.size[0] + 2 * pad, im.size[1] + 2 * pad + title_h), (255, 255, 255))
    panel_img.paste(im, (pad, pad + title_h))

    d = ImageDraw.Draw(panel_img)
    font_label = load_font(30, bold=True)
    font_title = load_font(22, bold=False)

    d.rectangle([pad, 10, pad + 42, 10 + 34], fill=(245, 245, 245), outline=(190, 190, 190), width=1)
    d.text((pad + 11, 9), label, fill=(40, 40, 40), font=font_label)
    d.text((pad + 56, 14), title, fill=(30, 30, 30), font=font_title)
    return panel_img


def main():
    for p in [A_PATH, B_PATH, C_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    a = Image.open(A_PATH).convert("RGB")
    b = Image.open(B_PATH).convert("RGB")
    c = Image.open(C_PATH).convert("RGB")

    # Layout inspired by MSF multi-panel pages: top row A/B, bottom row C full width
    top_w_each = 980
    c_w = top_w_each * 2 + 36

    pa = panel(a, "A", "Overall Metrics Comparison", pad=12, border=1, width=top_w_each)
    pb = panel(b, "B", "Activity Cliff Delta Comparison", pad=12, border=1, width=top_w_each)
    pc = panel(c, "C", "Stratified Robustness (F1)", pad=12, border=1, width=c_w)

    gap = 18
    canvas_w = pa.size[0] + pb.size[0] + gap
    top_h = max(pa.size[1], pb.size[1])
    canvas_h = top_h + gap + pc.size[1] + 40

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # top row
    canvas.paste(pa, (0, 0))
    canvas.paste(pb, (pa.size[0] + gap, 0))

    # bottom centered
    x_c = (canvas_w - pc.size[0]) // 2
    y_c = top_h + gap
    canvas.paste(pc, (x_c, y_c))

    # subtle outer frame
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=(210, 210, 210), width=1)

    out_png = os.path.join(OUT_DIR, "figure_3_ABC_msf_style.png")
    out_tif = os.path.join(OUT_DIR, "figure_3_ABC_msf_style.tif")
    out_pdf = os.path.join(OUT_DIR, "figure_3_ABC_msf_style.pdf")

    canvas.save(out_png, dpi=(300, 300))
    canvas.save(out_tif, dpi=(300, 300), compression="tiff_lzw")
    canvas.save(out_pdf, resolution=300.0)

    caption = os.path.join(OUT_DIR, "figure_3_ABC_caption.md")
    with open(caption, "w", encoding="utf-8") as f:
        f.write(
            "Figure 3. Integrated evaluation of TV-STFN under performance, cliff sensitivity, and physicochemical stratification. "
            "(A) Overall metric comparison between tv5overnight_1 and hard_v1_overnight. "
            "(B) True versus predicted permeability deltas for top-10 activity-cliff pairs. "
            "(C) F1 trends across MW and TPSA bins, highlighting degradation in highly complex regions.\n"
        )

    print(out_png)
    print(out_tif)
    print(out_pdf)
    print(caption)


if __name__ == "__main__":
    main()
