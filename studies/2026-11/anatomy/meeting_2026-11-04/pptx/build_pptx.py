"""Build the 2026-11-04 meeting deck on the user's Meeting_01.11 template.

The template is only read: it is copied in memory, its slides are dropped, and
the deck is written to OUT. Text runs are white through the theme (bg1), as on
the template's own slides; fonts come from the theme (Aptos / Aptos Display).
Speaker notes are the <aside> texts of the artifact deck's slides.
"""
import html
import json
import re
import sys
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

TEMPLATE = "/Users/miler/Promotion/Meetings with Luciano/Meeting_01.11.pptx"
HERE = Path(__file__).resolve().parent          # .../meeting_2026-11-04/pptx
FIG = HERE.parent                               # the meeting figures
SL = FIG / "deck" / "project" / "slides"        # the artifact deck's slides
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIG / "Meeting_04.11.pptx"

OR, LAV, BLU, GRN = "F0A35E", "D3C6EC", "8FC8F0", "7FD8B4"
X0, XW = 0.917, 11.5             # the template's title column
TITLE_PT = 40

SUB = re.compile(r"([A-Za-z])_([a-z0-9]+)")


def notes_and_alts(sid):
    x = (SL / f"{sid}.html").read_text()
    m = re.search(r"<aside>(.*?)</aside>", x, re.S)
    note = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip() if m else ""
    alts = [html.unescape(a) for a in re.findall(r'<img[^>]*alt="([^"]*)"', x) if a]
    alts += [html.unescape(a) for a in re.findall(r'aria-label="([^"]*)"', x)]
    return note, alts


def P(size, *segs, after=0, before=0, align=None, bold=False, line=None):
    """One paragraph: segments are plain strings (white) or (text, colour[, bold])."""
    out = []
    for s in segs:
        if isinstance(s, str):
            s = (s, None)
        out.append((s[0], s[1], s[2] if len(s) > 2 else bold))
    return dict(size=size, segs=out, after=after, before=before, align=align, line=line)


def _run(p, text, size, color, bold, sub=False):
    if not text:
        return
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    if color is None:
        f.color.theme_color = MSO_THEME_COLOR.BACKGROUND_1
    else:
        f.color.rgb = RGBColor.from_string(color)
    if sub:
        f._rPr.set("baseline", "-25000")


def add_runs(p, text, size, color, bold):
    """B_t, v_x ... become a letter with a real subscript."""
    pos = 0
    for m in SUB.finditer(text):
        _run(p, text[pos:m.start(2) - 1], size, color, bold)
        _run(p, m.group(2), size, color, bold, sub=True)
        pos = m.end()
    _run(p, text[pos:], size, color, bold)


def fill_frame(tf, paras, align=PP_ALIGN.LEFT):
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = para["align"] or align
        if para["after"]:
            p.space_after = Pt(para["after"])
        if para["before"]:
            p.space_before = Pt(para["before"])
        if para["line"]:
            p.line_spacing = para["line"]
        for text, color, bold in para["segs"]:
            add_runs(p, text, para["size"], color, bold)


def tbox(s, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, inset=0.05, name=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(inset)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    fill_frame(tf, paras, align)
    return tb


def title(s, text, size=TITLE_PT):
    tf = s.shapes.title.text_frame
    tf.text = ""
    p = tf.paragraphs[0]
    _run(p, text, size, None, False)


def pic(s, path, x, y, w=None, h=None, alt="", name=None):
    im = Image.open(path)
    ar = im.width / im.height
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    p = s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    p._element.nvPicPr.cNvPr.set("descr", alt)
    if name:
        p.name = name
    return p, w, h


def set_alpha_fill(sh, rgb, alpha):
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor.from_string(rgb)
    clr = sh._element.spPr.find(qn("a:solidFill"))[0]
    etree.SubElement(clr, qn("a:alpha")).set("val", str(int(round(alpha * 100000))))


def tile(s, x, y, w, h, rgb="FFFFFF", alpha=0.08, radius=0.06, name=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    set_alpha_fill(sh, rgb, alpha)
    sh.line.fill.background()
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- the deck
prs = Presentation(TEMPLATE)
sld_ids = prs.slides._sldIdLst
for sld in list(sld_ids):
    rid = sld.get(qn("r:id"))
    sld_ids.remove(sld)
    prs.part.drop_rel(rid)

L_TITLE, L_CONTENT, L_ONLY = prs.slide_layouts[0], prs.slide_layouts[1], prs.slide_layouts[5]
NOTE = {}
ALT = {}
for sid in json.loads((SL.parent / "deck.json").read_text())["order"]:
    NOTE[sid], ALT[sid] = notes_and_alts(sid)
# the notes in the words the slides now use (named structures, "shear layer")
for sid, a, b in [
        ("v2helps", "Losses at the jet tip and in the roll-up's inner ring; the alternating rings are features that moved.",
         "Losses at the tip of the dense tongue and in the inner ring of the low-density pocket; the alternating "
         "blue-red rings are features that sit at slightly different places in the two runs."),
        ("v2census", "along the shear lines", "along the shear layers"),
        ("v4where", "along the diagonal; its tip is the jet head.", "along the diagonal."),
        ("v4where", "a low-density pocket forms (the roll-up).", "a low-density pocket forms."),
        ("v3regimes", "in the 0.99c sliding flow", "along the 0.99c shear layers"),
        ("v2next", "local to the shear lines", "local to the shear layers")]:
    assert NOTE[sid].count(a) == 1, (sid, a)
    NOTE[sid] = NOTE[sid].replace(a, b)
for sid, n in NOTE.items():
    for word in ("roll-up", "jet head", "shear line", "sliding"):
        if word in n:
            print(f"note {sid}: contains '{word}'")


def new(layout=L_ONLY, sid=None, head=None):
    s = prs.slides.add_slide(layout)
    if head:
        title(s, head)
    if sid:
        notes(s, NOTE[sid])
    return s


# 1 cover
s = new(L_TITLE, "v2cover")
tf = s.shapes.title.text_frame
tf.text = ""
_run(tf.paragraphs[0], "Wed Nov 04", 60, None, False)
sub = s.placeholders[1].text_frame
sub.text = ""
fill_frame(sub, [P(24, "Meeting with Luciano"),
                 P(18, ("Exact RMHD — the four-quadrant problem at 128²: what works, what fails, and why", LAV),
                   before=14)], align=PP_ALIGN.CENTER)

# 2 topics
s = new(L_CONTENT, "v2topics", "Topics")
body = s.placeholders[1].text_frame
body.text = ""
fill_frame(body, [P(28, t, after=6) for t in ("What we have", "What fails", "Why it fails", "What would fix it")])
for p in body.paragraphs:
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.55)))
    pPr.set("indent", str(-Inches(0.55)))
    etree.SubElement(etree.SubElement(pPr, qn("a:buClr")), qn("a:schemeClr")).set("val", "bg1")
    etree.SubElement(pPr, qn("a:buAutoNum")).set("type", "arabicPeriod")

# 3 initial data
s = new(L_ONLY, "v4id", "The four-quadrant problem: initial data")
cw, ch, gap, gy = 2.6, 1.95, 0.1, 2.15
quads = [  # (col, row, alpha, line 1, line 2, colour of line 2)
    (0, 0, 0.14, "ρ = 0.1   p = 1", "v_x = 0.99  →", OR),
    (1, 0, 0.06, "ρ = 0.1   p = 0.01", "at rest", LAV),
    (0, 1, 0.30, "ρ = 0.5   p = 1", "at rest", LAV),
    (1, 1, 0.14, "ρ = 0.1   p = 1", "v_y = 0.99  ↑", OR),
]
for c, r, a, l1, l2, c2 in quads:
    x, y = X0 + c * (cw + gap), gy + r * (ch + gap)
    tile(s, x, y, cw, ch, alpha=a, radius=0.05, name=f"Quadrant {r * 2 + c + 1}")
    tbox(s, x, y, cw, ch, [P(20, l1, after=6), P(20, (l2, c2))], anchor=MSO_ANCHOR.MIDDLE,
         align=PP_ALIGN.CENTER)
xr = X0 + 2 * cw + gap + 0.55
tbox(s, xr, gy, X0 + XW - xr, 2 * ch + gap, [
    P(20, "Uniform field B\u00a0=\u00a00.5 at 45°", after=4),
    P(20, "B_x\u00a0=\u00a0B_y\u00a0=\u00a00.354,\u00a0 B_z\u00a0=\u00a0v_z\u00a0=\u00a00", after=16),
    P(20, "Ideal gas, Γ\u00a0=\u00a05/3", after=4),
    P(20, "Domain [−0.5,\u00a00.5]², outflow, t\u00a0=\u00a00.4", after=16),
    P(20, ("Two ", LAV), ("shear layers", OR),
      (" (left, bottom): a 0.99c quadrant slides past the dense one. Two shocks (top, right): "
       "the fast quadrants run into the low-pressure one", LAV)),
], anchor=MSO_ANCHOR.MIDDLE)

# 4 what we have
s = new(L_ONLY, "v2have", "Exact flux vs HLLD at 128²")
_, w, h = pic(s, FIG / "01_l1_vs_n.png", X0, 2.05, h=4.6, alt=ALT["v2have"][0])
xr = X0 + w + 0.5
tbox(s, xr, 2.05, X0 + XW - xr, 4.6, [
    P(88, ("−5.8%", OR, True), after=10, line=0.9),
    P(26, "density error against HLLD on the same grid", after=14),
    P(20, ("as good as HLLD at about 145², at 28× its wall time", LAV)),
], anchor=MSO_ANCHOR.MIDDLE)

# 5 where the exact flux helps
s = new(L_ONLY, "v2helps", "Where the exact flux helps")
_, w, h = pic(s, FIG / "02_err_maps_flux_128.png", X0, 2.0, w=XW, alt=ALT["v2helps"][0])
tbox(s, X0, 2.0 + h + 0.15, XW, 0.95, [
    P(22, "Closer than HLLD in half the cells — ",
      ("gains in the low-density pocket where the shear layers meet, and along the shear layers", OR)),
])

# 6 networks
s = new(L_ONLY, "v2nets", "Retrained networks win on their own problem")
rows = [("problem", "production network", "retrained on it"),
        ("Rotor", "94.2 / 78.4", "96.6 / 85.5"),
        ("Orszag–Tang", "97.0 / 86.2", "98.5 / 94.4"),
        ("Four-quadrant", "87.8 / 90.5", "91.9 / 97.4")]
gf = s.shapes.add_table(4, 3, Inches(X0), Inches(2.2), Inches(10.5), Inches(4 * 0.62))
gf.name = "Networks table"
tbl = gf.table
for j, wd in enumerate((3.9, 3.3, 3.3)):
    tbl.columns[j].width = Inches(wd)
for i, row in enumerate(rows):
    tbl.rows[i].height = Inches(0.62)
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = val
        r.font.size = Pt(22)
        if i > 0 and j == 2:
            r.font.bold = True
tbox(s, X0, 2.2 + 4 * 0.62 + 0.35, 10.5, 1.0, [
    P(20, ("% of held-out faces solved with one Newton attempt (all faces / field-rotating faces). "
           "None transfers to the other problems → one network per equation of state.", LAV)),
])

# 7 what fails
s = new(L_ONLY, "v2fails", "What fails")
tbox(s, X0, 2.0, XW, 4.7, [
    P(120, ("0.8%", OR, True), line=0.9, after=6),
    P(32, "of the attempted faces are lost to HLLD", after=16),
    P(22, ("37,940 of 4.71 million — every face traced to where and when it was lost", LAV)),
], anchor=MSO_ANCHOR.MIDDLE)

# 8 census
s = new(L_ONLY, "v2census", "What answers each face, step by step")
_, w, h = pic(s, FIG / "03_timeline.png", X0, 2.0, w=XW, alt=ALT["v2census"][0])
tbox(s, X0, 2.0 + h + 0.15, XW, 0.6, [
    P(20, "seven-wave 90%\u00a0· ", ("planar rescue 8.4%", BLU), "\u00a0· retry 0.7%\u00a0· four-unknown 0.5%\u00a0· ",
      ("lost 0.8%", OR)),
])

# 9 where the faces are lost
s = new(L_ONLY, "v4where", "Where the faces are lost")
_, w, h = pic(s, FIG / "c_structures_v2.png", X0, 1.95, h=5.0, alt=ALT["v4where"][0])
xr = X0 + w + 0.5
tbox(s, xr, 1.95, X0 + XW - xr, 5.0, [
    P(22, ("● A (52%)", OR, True), after=4),
    P(20, "along the two shear layers, and the edges of the dense tongue", after=22),
    P(22, ("● B, weak field (34%)", BLU, True), after=4),
    P(20, "around the low-density pocket where the shear layers meet", after=22),
    P(22, ("● B, strong field (14%)", GRN, True), after=4),
    P(20, "at the tip of the tongue, and where the shear layers end"),
], anchor=MSO_ANCHOR.MIDDLE)

# 10 ledger
s = new(L_ONLY, "v2ledger", "What we tried on the lost faces")
tbox(s, X0, 2.0, XW, 4.7, [
    P(120, ("83%", OR, True), line=0.9, after=6),
    P(32, "of the lost faces have no root that any of our methods finds", after=16),
    P(22, ("tried: more seeds and retries\u00a0· every planar branch\u00a0· continuation\u00a0· "
           "intermediate shocks\u00a0· compound waves with a field flip", LAV)),
], anchor=MSO_ANCHOR.MIDDLE)

# 11 two regimes
s = new(L_ONLY, "v3regimes", "Two situations explain every rescue and loss")
tw, th, ty = 5.6, 3.95, 1.95
for k, (head, col, img, alt, line1, line2) in enumerate([
        ("A — two waves at the same speed", OR, "schem_fan.png", ALT["v3regimes"][0],
         "The slow and the Alfvén wave travel within 0.1% of each other",
         "2.2M faces · 0.9% lost · 14% need the planar rescue"),
        ("B — field straight through the face", BLU, "schem_face.png", ALT["v3regimes"][1],
         "The field points almost straight through the face: hardly any tangential field",
         "0.5M faces · 3.4% lost · 15% planar · 5% four-unknown")]):
    x = X0 + k * (tw + 0.3)
    tile(s, x, ty, tw, th, name=f"Tile {'AB'[k]}")
    tbox(s, x + 0.25, ty + 0.15, tw - 0.5, 0.55, [P(22, (head, col, True))])
    pic(s, HERE / img, x + (tw - 3.9) / 2, ty + 0.75, w=3.9, alt=alt)
    tbox(s, x + 0.25, ty + 2.15, tw - 0.5, th - 2.25, [P(18, line1, after=8), P(16, (line2, LAV))])
tbox(s, X0, ty + th + 0.2, XW, 0.6, [P(22, "Everything else — ", ("42% of all faces — is never lost", OR))])

# 12 planar rescue
s = new(L_ONLY, "v2planar", "Why the planar rescue is needed")
w = 10.6
_, w, h = pic(s, FIG / "c_mechanism.png", (13.333 - w) / 2, 1.95, w=w, alt=ALT["v2planar"][0])
tbox(s, X0, 1.95 + h + 0.15, XW, 0.9, [
    P(22, "A slow shock riding the Alfvén speed makes the seven-wave system singular — ",
      ("the formulation, not the seed", OR)),
])

# 13, 14 loss types
for sid, head, img, paras in [
        ("v2typeA", "Loss type A: two slow waves on the face", "c_loss_A.png", [
            P(22, "Flow along the face at 0.987c", after=18),
            P(22, "Slow wave, Alfvén wave and contact all within 0.02 of x/t = 0", after=18),
            P(22, ("B_t changes twice there: a compound wave (shock + fan), outside the seven-wave family", OR))]),
        ("v2typeB", "Loss type B: field nearly normal", "c_loss_B.png", [
            P(22, ("Weak field: ", BLU, True), "B_t reverses across one jump — an intermediate shock", after=22),
            P(22, ("Strong field: ", GRN, True),
              "the slow wave moves pressure, not B_t — the variable the solver steers by")])]:
    s = new(L_ONLY, sid, head)
    _, w, h = pic(s, FIG / img, X0, 1.95, h=5.0, alt=ALT[sid][0])
    xr = X0 + w + 0.45
    tbox(s, xr, 1.95, X0 + XW - xr, 5.0, paras, anchor=MSO_ANCHOR.MIDDLE)

# 15 what would fix it
s = new(L_ONLY, "v2next", "What would fix it")
for k, (lab, col, text) in enumerate([
        ("A · 52%", OR, "a compound wave without a field flip: slow shock + attached fan"),
        ("B weak · 34%", BLU, "intermediate shocks on both sides of the face"),
        ("B strong · 14%", GRN, "slow waves steered by pressure on every face")]):
    y = 2.2 + k * 1.0
    pill = tile(s, X0, y, 2.6, 0.66, rgb=col, alpha=0.22, radius=0.5, name=f"Label {lab}")
    tf = pill.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    fill_frame(tf, [P(20, (lab, col, True))], align=PP_ALIGN.CENTER)
    tbox(s, X0 + 2.95, y, XW - 2.95, 0.66, [P(22, text)], anchor=MSO_ANCHOR.MIDDLE)
tbox(s, X0, 5.55, XW, 0.6, [P(22, ("And for speed: send the shear faces straight to the planar solver", LAV))])

# 16 intermediate shocks and the type-A fix
s = new(L_ONLY, "v4fixA", "Intermediate shocks, and the fix for type A")
tw, th, ty = 5.6, 4.5, 1.9
for k, (head, col, img, alt, line1, line2) in enumerate([
        ("Intermediate shock", BLU, "schem_is.png", ALT["v4fixA"][0],
         "The tangential field flips sign across one jump — possible only if the flow crosses the "
         "Alfvén speed inside the shock",
         "It conserves everything, but ideal MHD discards it: the slightest out-of-plane twist of the "
         "field splits it into an Alfvén wave and a slow shock. Our flow is exactly planar, so nothing "
         "twists it and the scheme keeps it — that is what the weak-field B faces form"),
        ("Type A: a compound wave, no flip", OR, "schem_compound.png", ALT["v4fixA"][1],
         "A slow shock moving exactly at the slow speed behind it, with a slow fan glued on — the field "
         "keeps its sign",
         "Our compound element builds only the flipping kind (Brio–Wu: an intermediate shock + fan); "
         "type A needs the non-flipping kind — one new piece in the planar solver, still three unknowns")]):
    x = X0 + k * (tw + 0.3)
    tile(s, x, ty, tw, th, name=f"Tile {k + 1}")
    tbox(s, x + 0.25, ty + 0.12, tw - 0.5, 0.5, [P(22, (head, col, True))])
    pic(s, HERE / img, x + (tw - 3.3) / 2, ty + 0.65, w=3.3, alt=alt)
    tbox(s, x + 0.25, ty + 1.85, tw - 0.5, th - 1.95, [P(16, line1, after=8), P(14, (line2, LAV))])
tbox(s, X0, ty + th + 0.12, XW, 0.75, [
    P(18, "Test: build it, replay the 335 type-A faces nothing solves today, and check solved faces for "
          "second solutions")])

# python-pptx relates the notes master it creates but does not list it in
# presentation.xml; Apple's renderers (QuickLook, Keynote) hang on that
pres = prs.part._element
if pres.find(qn("p:notesMasterIdLst")) is None:
    rid = next(r.rId for r in prs.part.rels.values() if r.reltype.endswith("/notesMaster"))
    lst = etree.Element(qn("p:notesMasterIdLst"))
    etree.SubElement(lst, qn("p:notesMasterId")).set(qn("r:id"), rid)
    pres.find(qn("p:sldMasterIdLst")).addnext(lst)

# the alt texts in the words the slides now use
for k, slide in enumerate(prs.slides):
    for sh in slide.shapes:
        if sh.shape_type == 13:  # picture
            c = sh._element.nvPicPr.cNvPr
            c.set("descr", c.get("descr", "").replace("shear lines", "shear layers"))

prs.save(OUT)
print("wrote", OUT, len(prs.slides._sldIdLst), "slides")
