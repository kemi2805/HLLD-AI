"""Mock-render a deck with the real Aptos metrics, and report text that does not fit.

There is no LibreOffice on this Mac and QuickLook substitutes Aptos, so this lays the
text out itself: greedy word wrap with Aptos / Aptos Bold, line height 1.22 em
(Aptos typo metrics: 1923 + 577 over 2048) times the paragraph's line spacing, plus
space before/after. Placeholder anchors and alignment follow the template's layouts
(title: middle, left; cover title: bottom, centred; cover subtitle: top, centred).
"""
import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.dml import MSO_COLOR_TYPE
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn

FD = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts/"
S = 120                       # px per inch
EMU = 914400
LH = 2500 / 2048              # Aptos single line height in em
_fonts = {}


def font(pt, bold):
    key = (round(pt * S / 72, 1), bold)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FD + ("Aptos-Bold.ttf" if bold else "Aptos.ttf"), key[0])
    return _fonts[key]


def px(emu):
    return emu / EMU * S


def color_of(run, default):
    c = run.font.color
    try:
        if c.type == MSO_COLOR_TYPE.RGB:
            return "#" + str(c.rgb)
        if c.type == MSO_COLOR_TYPE.SCHEME:
            return "#FFFFFF" if "BACKGROUND_1" in str(c.theme_color) else "#000000"
    except AttributeError:
        pass
    return default


def runs_of(par, default_pt, default_col, default_bold=False):
    out = []
    for r in par.runs:
        pt = r.font.size.pt if r.font.size else default_pt
        sub = r.font._rPr.get("baseline") is not None if r.font._rPr is not None else False
        bold = r.font.bold if r.font.bold is not None else default_bold
        out.append((r.text, pt * (0.66 if sub else 1.0), bool(bold), color_of(r, default_col), sub, pt))
    return out


def layout_par(par, width, default_pt, default_col, prefix="", default_bold=False):
    """-> list of lines; a line = (list of (text, pt, bold, col, sub), height px, ascent px)."""
    toks = []
    if prefix:
        toks.append((prefix, default_pt, default_bold, default_col, False, default_pt))
    for text, pt, bold, col, sub, base in runs_of(par, default_pt, default_col, default_bold):
        parts = text.split(" ")
        for i, w in enumerate(parts):
            t = w + (" " if i < len(parts) - 1 else "")
            if t:
                toks.append((t, pt, bold, col, sub, base))
    lines, cur, cw = [], [], 0.0
    for tok in toks:
        tw = font(tok[1], tok[2]).getlength(tok[0])
        if cur and cw + font(tok[1], tok[2]).getlength(tok[0].rstrip()) > width:
            lines.append(cur)
            cur, cw = [], 0.0
        cur.append(tok)
        cw += tw
    lines.append(cur)
    lsp = par.line_spacing if isinstance(par.line_spacing, float) else 1.0
    out = []
    for ln in lines:
        base = max([t[5] for t in ln] or [default_pt])
        out.append((ln, base * S / 72 * LH * lsp, base * S / 72 * 0.94 * lsp))
    return out


def frame(draw, tf, x, y, w, h, anchor, align, default_pt, default_col, name, report, numbered=False,
          default_bold=False):
    l, r = px(tf.margin_left), px(tf.margin_right)
    t, b = px(tf.margin_top), px(tf.margin_bottom)
    width = w - l - r
    blocks, total = [], 0.0
    for i, par in enumerate(tf.paragraphs):
        sb = par.space_before.pt * S / 72 if par.space_before is not None and i > 0 else 0
        sa = par.space_after.pt * S / 72 if par.space_after is not None and i < len(tf.paragraphs) - 1 else 0
        lines = layout_par(par, width, default_pt, default_col, f"{i + 1}.  " if numbered else "", default_bold)
        hgt = sum(ln[1] for ln in lines)
        blocks.append((par, lines, sb, sa))
        total += sb + hgt + sa
        for ln in lines:
            lw = sum(font(tk[1], tk[2]).getlength(tk[0]) for tk in ln[0])
            if lw > width + 1 and len(ln[0]) == 1:
                report.append(f"{name}: a word is wider than the box ({lw / S:.2f} > {width / S:.2f} in)")
    avail = h - t - b
    if total > avail + 0.5:
        report.append(f"{name}: text needs {total / S:.2f} in, box has {avail / S:.2f} in")
    yy = y + t + {MSO_ANCHOR.MIDDLE: (avail - total) / 2, MSO_ANCHOR.BOTTOM: avail - total}.get(anchor, 0)
    for par, lines, sb, sa in blocks:
        yy += sb
        al = par.alignment if par.alignment is not None else align
        for toks, lh, asc in lines:
            lw = sum(font(tk[1], tk[2]).getlength(tk[0]) for tk in toks)
            xx = x + l + {PP_ALIGN.CENTER: (width - lw) / 2, PP_ALIGN.RIGHT: width - lw}.get(al, 0)
            for text, pt, bold, col, sub, base in toks:
                f = font(pt, bold)
                fa = f.getmetrics()[0]
                draw.text((xx, yy + asc - fa + (0.12 * base * S / 72 if sub else 0)), text, font=f, fill=col)
                xx += f.getlength(text)
            yy += lh
        yy += sa
    return total


def render(deck, outdir, bg_png):
    prs = Presentation(deck)
    W, H = int(prs.slide_width / EMU * S), int(prs.slide_height / EMU * S)
    bg = Image.open(bg_png).convert("RGB").resize((W, int(H * 1.01)))
    bg = bg.crop((0, 0, W, H))
    outdir.mkdir(parents=True, exist_ok=True)
    report, paths = [], []
    for k, slide in enumerate(prs.slides, 1):
        im = bg.copy().convert("RGBA")
        for sh in slide.shapes:
            x, y, w, h = px(sh.left), px(sh.top), px(sh.width), px(sh.height)
            name = f"slide {k} / {sh.name}"
            if x < -1 or y < -1 or x + w > W + 1 or y + h > H + 1:
                report.append(f"{name}: outside the slide")
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pim = Image.open(io.BytesIO(sh.image.blob)).convert("RGBA").resize((round(w), round(h)))
                im.alpha_composite(pim, (round(x), round(y)))
                continue
            if sh.has_table if hasattr(sh, "has_table") else False:
                tbl = sh.table
                yy = y
                for i, row in enumerate(tbl.rows):
                    xx = x
                    rh = px(row.height)
                    for j, col in enumerate(tbl.columns):
                        cwid = px(col.width)
                        fill = "#156082" if i == 0 else ("#A1BFCD" if i % 2 else "#D0DFE6")
                        ImageDraw.Draw(im).rectangle([xx, yy, xx + cwid - 2, yy + rh - 2], fill=fill)
                        cell = tbl.cell(i, j)
                        frame(ImageDraw.Draw(im), cell.text_frame, xx, yy, cwid, rh, MSO_ANCHOR.MIDDLE,
                              PP_ALIGN.LEFT, 18, "#FFFFFF" if i == 0 else "#000000", f"{name} cell {i},{j}",
                              report, default_bold=(i == 0))
                        xx += cwid
                    yy += rh
                continue
            if sh.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE:
                sf = sh._element.spPr.find(qn("a:solidFill"))
                if sf is not None:
                    clr = sf[0]
                    a = clr.find(qn("a:alpha"))
                    alpha = int(a.get("val")) / 100000 if a is not None else 1.0
                    rgb = tuple(int(clr.get("val")[i:i + 2], 16) for i in (0, 2, 4))
                    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
                    rad = sh.adjustments[0] * min(w, h) if len(sh.adjustments) else 0
                    ImageDraw.Draw(ov).rounded_rectangle([x, y, x + w, y + h], radius=rad,
                                                         fill=rgb + (int(255 * alpha),))
                    im.alpha_composite(ov)
            if not sh.has_text_frame or not sh.text_frame.text.strip():
                continue
            tf = sh.text_frame
            anchor, align, dpt = tf.vertical_anchor, PP_ALIGN.LEFT, 18
            numbered = False
            if sh.is_placeholder:
                pt = sh.placeholder_format.type
                if pt == PP_PLACEHOLDER.CENTER_TITLE:
                    anchor, align, dpt = MSO_ANCHOR.BOTTOM, PP_ALIGN.CENTER, 60
                elif pt == PP_PLACEHOLDER.SUBTITLE:
                    anchor, align, dpt = MSO_ANCHOR.TOP, PP_ALIGN.CENTER, 24
                elif pt == PP_PLACEHOLDER.TITLE:
                    anchor, align, dpt = MSO_ANCHOR.MIDDLE, PP_ALIGN.LEFT, 44
                else:
                    anchor, dpt, numbered = MSO_ANCHOR.TOP, 28, True
            frame(ImageDraw.Draw(im), tf, x, y, w, h, anchor, align, dpt, "#000000", name, report, numbered)
        p = outdir / f"slide{k:02d}.png"
        im.convert("RGB").save(p)
        paths.append(p)
    return report, paths


if __name__ == "__main__":
    deck, outdir, bg = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    report, paths = render(deck, outdir, bg)
    print("\n".join(report) if report else "every text frame fits")
    # contact sheets of 4 slides each, half size
    for g in range(0, len(paths), 4):
        ims = [Image.open(p) for p in paths[g:g + 4]]
        w, h = ims[0].width // 2, ims[0].height // 2
        sheet = Image.new("RGB", (2 * w + 10, 2 * h + 10), "white")
        for i, im in enumerate(ims):
            sheet.paste(im.resize((w, h)), ((i % 2) * (w + 10), (i // 2) * (h + 10)))
        sheet.save(outdir / f"sheet{g // 4 + 1}.png")
