#!/usr/bin/env python3
"""Academic 16:9 deck from supervisor-fang-brief.canvas.tsx.

Writes docs/pc2/2026-09-08-autosa-mm-supervisor-brief.pptx
Tile sizes from autosa-mm-rank1-tiles / rank1-match-transformations /
2026-09-06 handoff. Module cycles from autosa_mm_csynth.rpt (PE16 20260904).
Never mix 4228/320 with 940/5344. Never say DSE.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import nsmap
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

OUT = Path(__file__).with_name("2026-09-08-autosa-mm-supervisor-brief.pptx")

NAVY = RGBColor(0x0F, 0x2C, 0x52)
NAVY2 = RGBColor(0x1B, 0x3A, 0x6B)
GOLD = RGBColor(0x8B, 0x6F, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x4A, 0x4A, 0x4A)
RULE = RGBColor(0xC8, 0xC4, 0xB8)
PALE = RGBColor(0xF4, 0xF1, 0xEA)
TILE = RGBColor(0xC5, 0xD4, 0xE8)
TILE2 = RGBColor(0xE8, 0xD9, 0xB8)
ROW_ALT = RGBColor(0xEE, 0xF1, 0xF6)
PASS = RGBColor(0x1F, 0x4E, 0x3D)
FAIL = RGBColor(0x8B, 0x1E, 0x1E)
BOX = RGBColor(0xE8, 0xEE, 0xF5)

W = Inches(13.333)
H = Inches(7.5)
ML = Inches(0.45)
MR = Inches(0.45)


def rgb_hex(c: RGBColor) -> str:
    return f"{c[0]:02X}{c[1]:02X}{c[2]:02X}"


def set_run(run, text, size=14, bold=False, color=INK, italic=False, font="Calibri"):
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def _ensure_run(p):
    if p.runs:
        return p.runs[0]
    return p.add_run()


def add_textbox(slide, l, t, w, h, text, *, size=14, bold=False, color=INK, align=PP_ALIGN.LEFT, italic=False, font="Calibri"):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(_ensure_run(p), text, size, bold, color, italic, font)
    return box


def tb(slide, l, t, w, h, lines, *, size=13, color=INK, bold_first=False, align=PP_ALIGN.LEFT, spacing=1.0):
    """lines: str or list of (text, kwargs) or str."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(4)
        if isinstance(item, tuple):
            text, kw = item
            set_run(_ensure_run(p), text, **kw)
        else:
            set_run(_ensure_run(p), item, size=size, bold=(bold_first and i == 0), color=color)
    return box


def rect(slide, l, t, w, h, fill, line=None, line_w=Pt(1.0)):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = line_w
    return sh


def labeled_box(slide, l, t, w, h, text, *, fill=BOX, line=NAVY, size=11, bold=True, tcolor=NAVY, align=PP_ALIGN.CENTER):
    sh = rect(slide, l, t, w, h, fill, line, Pt(1.25))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.04)
    tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = align
    set_run(_ensure_run(p), text, size=size, bold=bold, color=tcolor)
    return sh


def arrow(slide, x1, y1, x2, y2, color=NAVY):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    return c


def footer(slide, n, total, extra="c2hls  ·  autosa_mm vs AutoSA rank-1  ·  U280 3.33 ns  ·  Vitis HLS 2023.2"):
    rect(slide, 0, Inches(7.22), W, Inches(0.28), NAVY)
    add_textbox(slide, ML, Inches(7.23), Inches(11.2), Inches(0.24), extra, size=10, color=WHITE)
    add_textbox(slide, Inches(12.15), Inches(7.23), Inches(0.9), Inches(0.24), f"{n} / {total}", size=10, color=WHITE, align=PP_ALIGN.RIGHT)


def header(slide, title, kicker=""):
    rect(slide, 0, 0, W, Inches(0.72), NAVY)
    rect(slide, 0, Inches(0.72), W, Inches(0.06), GOLD)
    if kicker:
        add_textbox(slide, ML, Inches(0.06), Inches(12.4), Inches(0.22), kicker, size=11, color=GOLD, bold=True)
        add_textbox(slide, ML, Inches(0.26), Inches(12.4), Inches(0.42), title, size=22, bold=True, color=WHITE, font="Calibri")
    else:
        add_textbox(slide, ML, Inches(0.16), Inches(12.4), Inches(0.48), title, size=22, bold=True, color=WHITE, font="Calibri")


def set_cell(cell, text, *, size=11, bold=False, fill=None, color=INK, align=PP_ALIGN.LEFT):
    cell.text = ""
    p = cell.text_frame.paragraphs[0]
    p.alignment = align
    cell.text_frame.word_wrap = True
    set_run(_ensure_run(p), text, size=size, bold=bold, color=color)
    if fill is not None:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        solid = etree.SubElement(tcPr, qn("a:solidFill"))
        srgb = etree.SubElement(solid, qn("a:srgbClr"))
        srgb.set("val", rgb_hex(fill))
    # margins
    for attr, val in (("marL", "60000"), ("marR", "60000"), ("marT", "40000"), ("marB", "40000")):
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.set(qn(f"a:{attr}") if False else attr, val)
    # python-pptx uses marL on tcPr in EMUs via XML
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.set("marL", "72000")
    tcPr.set("marR", "72000")
    tcPr.set("marT", "36000")
    tcPr.set("marB", "36000")


def add_table(slide, left, top, width, rows, col_w, *, font=11, header=True):
    """rows: list of list of str. First row header if header=True."""
    n_r = len(rows)
    n_c = len(rows[0])
    table = slide.shapes.add_table(n_r, n_c, left, top, width, Inches(0.28 * n_r + 0.08)).table
    for i, w in enumerate(col_w):
        table.columns[i].width = w
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            is_h = header and r == 0
            alt = (not is_h) and (r % 2 == 0)
            fill = NAVY if is_h else (ROW_ALT if alt else WHITE)
            color = WHITE if is_h else INK
            al = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            # numeric-ish last cols stay center
            set_cell(cell=table.cell(r, c), text=str(val), size=font, bold=is_h, fill=fill, color=color, align=al)
    return table


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def new_prs():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def s_title(prs, n, total):
    s = blank(prs)
    rect(s, 0, 0, W, H, NAVY)
    rect(s, 0, Inches(5.85), W, Inches(1.65), NAVY2)
    rect(s, ML, Inches(1.35), Inches(1.6), Inches(0.07), GOLD)
    add_textbox(s, ML, Inches(1.55), Inches(12.4), Inches(1.4),
                "Closing the AutoSA matrix-multiply gap\nwith an LLM HLS agent",
                size=32, bold=True, color=WHITE, font="Calibri")
    add_textbox(s, ML, Inches(3.35), Inches(12.4), Inches(1.5),
                "Kernel autosa_mm  (I = J = K = 64, float)   ·   Alveo U280, 3.33 ns, Vitis HLS 2023.2\n"
                "Reference: AutoSA rank-1 HLS (paper Fig. 11a Design 1 class), not ChatHLS\n"
                "Agent: DeepSeek-v4-flash   ·   seed: plain.cpp, not stripped kernel0",
                size=16, color=RGBColor(0xD0, 0xD8, 0xE8))
    add_textbox(s, ML, Inches(6.05), Inches(12.4), Inches(1.1),
                "Supervisor briefing  ·  answers to the requested study  ·  8 September 2026\n"
                "Metrics: latency min, latency max, DSP. Interval and HLS estimated clock are not comparison metrics.\n"
                "Middle stage is a 16×4 compute rewrite (hide load-store), not AutoSA design-space search.",
                size=13, color=RGBColor(0xC5, 0xD0, 0xE0))
    notes(s, "Open with two comparisons. Do not put 940 next to 4228. Do not say DSE.")
    return s


def s_protocol(prs, n, total):
    s = blank(prs)
    header(s, "Experimental protocol", "Held fixed for the study")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.05), Inches(12.4),
        [
            ["Item", "Value"],
            ["Kernel / size", "autosa_mm, I = J = K = 64, typedef float data_t"],
            ["ABI", "extern \"C\" void autosa_mm(A[I][K], B[J][K], C[I][J])  —  B stored B[j][k]"],
            ["Gold", "C = A · B from zero. Kernel must not treat DRAM C as an accumulate input."],
            ["Device / tool", "xcu280-fsvh2892-2L-e  ·  Vitis HLS 2023.2  ·  clock 3.33 ns"],
            ["Seed", "related_work/.../autosa_mm/plain.cpp. Not stripped AutoSA kernel0."],
            ["Reference", "AutoSA rank-1 4228 cycles / 320 DSP (Design 1, 1-D along i). Not ChatHLS."],
            ["Agent", "DeepSeek-v4-flash. Cosim off except where named."],
            ["Reported metrics", "Latency min, latency max, DSP. Do not quote interval or estimated clock."],
        ],
        [Inches(2.6), Inches(9.8)],
        font=13,
    )
    notes(s, "Trap: an older plainseed_stream run generated AutoSA rank-1 and stripped pragmas. This study does not use that seed.")


def s_two_comparisons(prs, n, total):
    s = blank(prs)
    header(s, "Two comparisons  —  do not mix them", "The questions were two")
    footer(s, n, total)
    # two cards
    rect(s, ML, Inches(1.1), Inches(6.05), Inches(5.7), PALE, RULE)
    rect(s, Inches(6.85), Inches(1.1), Inches(6.05), Inches(5.7), PALE, RULE)
    labeled_box(s, Inches(0.6), Inches(1.25), Inches(5.75), Inches(0.42),
                "Comparison 1  ·  iso-compute", fill=NAVY, line=NAVY, size=16, tcolor=WHITE)
    labeled_box(s, Inches(7.0), Inches(1.25), Inches(5.75), Inches(0.42),
                "Comparison 2  ·  spend DSP", fill=NAVY, line=NAVY, size=16, tcolor=WHITE)
    tb(s, Inches(0.7), Inches(1.85), Inches(5.6), Inches(4.6), [
        ("16 PE × SIMD 4  ≈  320 DSP", {"size": 16, "bold": True, "color": NAVY}),
        ("Same MAC budget as AutoSA rank-1.", {"size": 14, "color": INK}),
        ("AutoSA rank-1:  4228 / 320", {"size": 18, "bold": True, "color": NAVY}),
        ("Agent stream (empty skills JSON):  4216 / 320", {"size": 14, "color": INK}),
        ("Agent stream (90-skill mmflow):  4292 / 320", {"size": 14, "color": INK}),
        ("Aug 18 first locked stream:  4285 / 320", {"size": 14, "color": INK}),
        ("Fair architecture-class comparison.", {"size": 13, "italic": True, "color": MUTED}),
        ("Pass band used later: ≤ ×1.02 (4312).", {"size": 13, "color": MUTED}),
    ])
    tb(s, Inches(7.1), Inches(1.85), Inches(5.6), Inches(4.6), [
        ("Flash allowed to spend U280 DSP (9024)", {"size": 16, "bold": True, "color": NAVY}),
        ("Not a fair beat of 4228. ~17× DSP.", {"size": 14, "color": INK}),
        ("Champion:  940 / 5344   (csynth)", {"size": 18, "bold": True, "color": NAVY}),
        ("Cosim PASS 1071. Kernel type: no.", {"size": 14, "color": INK}),
        ("Do not put 940 or 871 next to 4228.", {"size": 14, "bold": True, "color": FAIL}),
        ("871 csynth / 1224 cosim is fused I/O,", {"size": 13, "color": INK}),
        ("not the champion and not this comparison.", {"size": 13, "color": MUTED}),
    ])
    notes(s, "If one slide mixes 940 and 4228 as a speedup, the rest of the talk is noise.")


def s_method(prs, n, total):
    s = blank(prs)
    header(s, "1. Original request: one kernel, gap vs rank-1", "17 August  ·  what was held, what was not")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Held fixed", "Not held fixed", "Not done"],
            [
                "Top name, ABI, B layout J×K, U280, 3.33 ns, gold-from-zero TB, iso-compute 16×4 on comparison 1",
                "Internal nest. Flash, compute rewrite, and stream each change the code. That is the method.",
                "AutoSA array_part / space_time / simd search. We did not call AutoSA. We did not emit kernel0 / A_t16 / L2 FIFOs as the agent result.",
            ],
        ],
        [Inches(4.13), Inches(4.13), Inches(4.14)],
        font=13,
    )
    tb(s, ML, Inches(3.15), Inches(12.4), Inches(3.8), [
        ("What AutoSA automates, cloned as a plan (not the compiler)", {"size": 16, "bold": True, "color": NAVY}),
        ("(1) Many PEs that do not wait on one running total — compute rewrite, PE=16 SIMD=4 via post_flash_pe_recipe.py.", {"size": 14}),
        ("(2) Memory filling the next chapter while they work — stream rewrite: DATAFLOW + hls::stream, A private per PE, B forwarded, C drained.", {"size": 14}),
        ("Knobs for this mm were fixed at 16×4. We did not hunt them on the locked slide.", {"size": 14}),
        ("Disk still labels the middle stage dse. Slides say compute rewrite / hide load-store.", {"size": 14, "italic": True, "color": MUTED}),
    ])


def s_pipeline_chart(prs, n, total):
    s = blank(prs)
    header(s, "Iso-compute path (comparison 1)", "Csynth latency min, cycles  ·  naive 2 445 313 omitted from the plot (destroys scale)")
    footer(s, n, total)
    data = CategoryChartData()
    data.categories = [
        "Flash 90-skill",
        "Compute rewrite",
        "Enforcement",
        "Stream +skills",
        "Stream no JSON",
        "AutoSA rank-1",
    ]
    data.add_series("Csynth latency min (cycles)", [139484, 13160, 12893, 4292, 4216, 4228])
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, ML, Inches(1.0), Inches(12.4), Inches(4.55), data).chart
    chart.has_legend = False
    plot = chart.plots[0]
    plot.has_data_labels = True
    try:
        plot.data_labels.font.size = Pt(9)
        plot.data_labels.font.color.rgb = INK
    except Exception:
        pass
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.size = Pt(10)
    add_textbox(s, ML, Inches(5.55), Inches(12.4), Inches(1.5),
                "Flash / compute / stream: frozen 20260830_mmflow. Enforcement: 20260829_123043.\n"
                "Rank-1: AutoSA 20260709_full21_dse resynth 3.33 ns. Aug 18 first locked stream (nav_n) was 4285 / 320; mmflow with-skills is 4292 / 320.\n"
                "Gap vs 4228: with-skills +64 cycles (+1.5%); empty-JSON stream −12 cycles (−0.3%). Wave1 pass band ≤ 4312.",
                size=13, color=MUTED)
    notes(s, "90-skill flash 139484 / 10 DSP is a legal kernel with almost no MAC parallelism. PE recipe + stream close rank-1.")


def s_stage_table(prs, n, total):
    s = blank(prs)
    header(s, "Pipeline stages (frozen mmflow unless noted)", "Comparison 1  ·  latency min–max and DSP")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Stage", "Job", "Latency min–max", "DSP", "Campaign"],
            ["Phase B naive", "Functional HLS, no opt pack", "2 445 313", "50", "hls_synth__ref_baseline"],
            ["Flash 90-skill", "Legal kernel + generic HLS pack", "139 484", "10", "20260830_mmflow"],
            ["Compute rewrite", "16×4 on-chip PEs, II=1 compute, LCST serial", "13 160", "352", "20260830_mmflow"],
            ["Enforcement", "Ping-pong DATAFLOW judged present", "12 893", "320", "20260829_123043"],
            ["Stream", "Hide load/store on that array", "4292", "320", "20260830_mmflow"],
            ["Stream, empty JSON", "Same PE recipe, skill_count=0", "4216", "320", "20260902_mmns"],
            ["AutoSA rank-1", "Published Design 1 class", "4228", "320", "AutoSA inventory"],
        ],
        [Inches(2.15), Inches(4.0), Inches(2.0), Inches(0.85), Inches(3.4)],
        font=12,
    )
    add_textbox(s, ML, Inches(6.35), Inches(12.4), Inches(0.7),
                "Attribution at 320 DSP: PE recipe + stream skeleton, not the 90-skill dump. Empty JSON still closes rank-1.",
                size=14, color=NAVY, bold=True)


def s_ablation(prs, n, total):
    s = blank(prs)
    header(s, "2. Ablation: generic HLS vs enforcement vs systolic rewrite", "Requested pack-level, not skill 3 of 5  ·  same ABI, csim on, cosim off")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Col", "What was given", "Latency", "DSP", "Result"],
            ["1", "Generic HLS (90-skill flash)", "139 484", "10", "Legal kernel, no PE array. Almost no MAC parallelism."],
            ["2", "Skills + ping-pong/DATAFLOW enforcement", "12 893", "320", "Pragma present. Finish still load+compute+store. Overlap gate fail."],
            ["3a", "Systolic: 90-skill + PE recipe + stream pack", "4292", "320", "+1.5% vs 4228."],
            ["3b", "Same PE recipe; skills JSON empty", "4216", "320", "−0.3% vs 4228."],
        ],
        [Inches(0.7), Inches(3.9), Inches(1.5), Inches(0.8), Inches(5.5)],
        font=13,
    )
    tb(s, ML, Inches(4.85), Inches(12.4), Inches(2.0), [
        ("The PE recipe plus stream skeleton matter more than the 90-skill dump. Generic flash can emit a legal kernel with 10 DSP. Empty JSON still closes rank-1 if the compute-rewrite/stream path still instantiates 16×4.", {"size": 15, "color": INK}),
        ("Column 1 is not a PE-free baseline of the systolic path: it is the 90-skill flash kernel. Column 3b is the systolic PE recipe with the skill JSON emptied.", {"size": 14, "color": MUTED}),
    ])


def s_flash_weak(prs, n, total):
    s = blank(prs)
    header(s, "3. Why flash was II=4 / low DSP, and why later stages sat on it", "Repair success = csim + legal csynth. No DSP / II judge until 3 September.")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Fact", "Number / mechanism"],
            ["Aug 18 flash", "149 082 cycles, 6 DSP, II=4 on k (scalar FP-add recurrence)"],
            ["Frozen mmflow flash", "139 484 cycles, 10 DSP"],
            ["Flash job definition", "Legal, testbench-correct HLS with interfaces. Success = csim pass + csynth."],
            ["Why repair did not catch II=4", "Repair loops on compile / csim / csynth failure, not on II or DSP. An II=4 kernel synthesizes."],
            ["Compute rewrite after that flash", "13 160 / 352. Compute nest ~4816 at II=1. Remaining: sequential load ~4100 + store ~4100."],
            ["Wrong comparison", "Measuring flash vs AutoSA 4228 is not iso-compute. Flash is not the PE array."],
        ],
        [Inches(3.4), Inches(9.0)],
        font=13,
    )
    add_textbox(s, ML, Inches(6.35), Inches(12.4), Inches(0.7),
                "Section 8 (spend DSP) answers “flash itself must stop under-using DSP.” Multi-PE replaces the nest; it does not require flash II=1.",
                size=14, color=NAVY, bold=True)


def s_tiling_rank1(prs, n, total):
    """Figure A: AutoSA rank-1 Design 1 tiles. Verified sizes."""
    s = blank(prs)
    header(s, "Tiling A. AutoSA rank-1  (Design 1 class)",
           "array_part[16, 64, 32]  ·  latency[1, 32]  ·  simd[4]  ·  1-D along i  ·  4228 / 320 DSP")
    footer(s, n, total, "Tile sizes from kernel0 / autosa-mm-rank1-tiles  ·  not candidate 9")

    # PE chain
    add_textbox(s, ML, Inches(0.92), Inches(12.4), Inches(0.28),
                "16 PEs in a chain (not a 2-D i,j or i,k mesh). PE p owns C row i0+p, all 64 columns, as local_C[64].",
                size=13, color=INK)
    pe_y = Inches(1.22)
    pe_w, pe_h = Inches(0.62), Inches(0.42)
    gap = Inches(0.08)
    x0 = ML
    for i in range(16):
        x = x0 + i * (pe_w + gap)
        labeled_box(s, x, pe_y, pe_w, pe_h, f"PE{i}", fill=TILE, line=NAVY, size=9, tcolor=NAVY)
        if i < 15:
            arrow(s, x + pe_w, pe_y + pe_h / 2, x + pe_w + gap, pe_y + pe_h / 2)
    add_textbox(s, ML, Inches(1.66), Inches(12.4), Inches(0.28),
                "B forwarded PE p → PE p+1 each beat.  SIMD-k = 4 independent a[s]·b[s], then one add into Crow. Reduction is the outer K-tile.",
                size=12, color=MUTED)

    # Three matrices
    def matrix_block(x, y, title, live, rest_note):
        labeled_box(s, x, y, Inches(3.9), Inches(0.32), title, fill=NAVY, line=NAVY, size=12, tcolor=WHITE)
        # outer 64x64
        labeled_box(s, x, y + Inches(0.36), Inches(3.9), Inches(2.15), "", fill=WHITE, line=NAVY, size=8)
        # live tile inside
        labeled_box(s, x + Inches(0.12), y + Inches(0.48), Inches(3.66), Inches(1.55), live,
                    fill=TILE, line=NAVY2, size=12, tcolor=NAVY)
        add_textbox(s, x, y + Inches(2.54), Inches(3.9), Inches(0.45), rest_note, size=11, color=MUTED, align=PP_ALIGN.CENTER)

    matrix_block(ML, Inches(2.05), "A  (I × K = 64 × 64)",
                 "Live this step:\n16 rows × 32 k\n(I-tile × K-tile)",
                 "4 I-tiles (c0 = 0…3). A unique from DRAM.")
    matrix_block(Inches(4.7), Inches(2.05), "B  (J × K, stored B[j][k])",
                 "Live this step:\n64 × 32  (full J × K-tile)\nping-pong on-chip",
                 "J is not coarse-tiled. B replay once per I-tile (4× DRAM B).")
    matrix_block(Inches(8.95), Inches(2.05), "C  (I × J = 64 × 64)",
                 "Live in the PE:\n16 rows × 64 cols\nlocal_C[64] per PE",
                 "Drain C only on last k-group of K-tile 1. Gold from zero: no load_C.")

    # Tile schedule
    add_table(
        s, ML, Inches(5.15), Inches(12.4),
        [
            ["Loop", "Tiles", "On-chip this step", "I/O schedule"],
            ["c0  (I)", "4 × 16 rows  (i = 0–15, 16–31, 32–47, 48–63)", "A 16×32; C 16×64", "A unique; B replayed each I-tile"],
            ["c2  (K)", "2 × 32 k  (k = 0–31 then 32–63)", "B 64×32 ping-pong", "Fill tile t+1 while PE consumes tile t (paper §6.3)"],
            ["J", "not coarse-tiled", "full C row of 64 in each PE", "Inner c6/c4 walk 32+32 j; drain after last K-tile"],
        ],
        [Inches(1.2), Inches(4.5), Inches(2.7), Inches(4.0)],
        font=12,
    )
    notes(s, "Candidate 9 is 2-D J×K tiles (array_part[64,32,32], ~1296 DSP). Do not present cand-9 as Design 1. Planned rank-1-shaped manual uses A 16×32, B 64×32, C 16×64 — different from the 940 champion.")


def s_tiling_stream(prs, n, total):
    """Figure B: agent iso-compute stream vs enforcement."""
    s = blank(prs)
    header(s, "Tiling B. Agent iso-compute stream vs enforcement",
           "Comparison 1  ·  same 16×4  ·  320 DSP  ·  not the 940 kernel")
    footer(s, n, total)

    # Left: stream
    rect(s, ML, Inches(0.95), Inches(6.15), Inches(6.05), PALE, RULE)
    labeled_box(s, Inches(0.55), Inches(1.05), Inches(5.95), Inches(0.38),
                "Hide load-store  ·  4292 / 4216 cycles  ·  320 DSP", fill=NAVY, line=NAVY, size=14, tcolor=WHITE)
    tb(s, Inches(0.6), Inches(1.5), Inches(5.85), Inches(1.35), [
        ("Same PE geometry as Design 1: 16 PE × SIMD 4, 1-D along i.", {"size": 13, "bold": True, "color": NAVY}),
        ("Frozen mmflow parked full B on an I-strip (not rank-1 K-tiles of 32). Still closes 4228 at 320 DSP.", {"size": 12, "color": INK}),
    ])
    # PE + FIFOs schematic
    for i in range(8):
        labeled_box(s, Inches(0.65 + i * 0.72), Inches(2.85), Inches(0.62), Inches(0.36), f"PE{i}", fill=TILE, line=NAVY, size=9)
    add_textbox(s, Inches(0.65), Inches(3.25), Inches(5.7), Inches(0.25), "… PE8–PE15  ·  A private per PE  ·  B forwarded  ·  C drained", size=11, color=MUTED)
    labeled_box(s, Inches(0.65), Inches(3.55), Inches(2.7), Inches(0.85), "Packed FIFOs\nap_uint<128>\nSIMD 4 floats / beat", fill=WHITE, line=NAVY, size=11)
    labeled_box(s, Inches(3.5), Inches(3.55), Inches(2.7), Inches(0.85), "Crow BIND_STORAGE\nram_2p  (not complete-\npartition on J)", fill=WHITE, line=NAVY, size=11)
    labeled_box(s, Inches(0.65), Inches(4.5), Inches(5.55), Inches(0.7), "Flatten pe_kj  1024 / 4096 beats\nFuse Crow init + drain into MAC", fill=WHITE, line=NAVY, size=12)
    add_textbox(s, Inches(0.65), Inches(5.3), Inches(5.85), Inches(1.4),
                "One run ≈ slowest room. Load, 16 PEs, and store run together.\n"
                "Failures taught as avoid-rules: struct-of-float FIFOs 16 878 / 80 DSP; Crow complete-partition J → mux II=4; flatten-off refill 7162.",
                size=12, color=INK)

    # Right: enforcement
    rect(s, Inches(6.8), Inches(0.95), Inches(6.1), Inches(6.05), PALE, RULE)
    labeled_box(s, Inches(6.9), Inches(1.05), Inches(5.9), Inches(0.38),
                "Enforcement DATAFLOW + ping-pong  ·  12 893 / 320", fill=FAIL, line=FAIL, size=13, tcolor=WHITE)
    tb(s, Inches(6.95), Inches(1.55), Inches(5.8), Inches(1.0), [
        ("Judge: ping-pong DATAFLOW present on code and csynth.", {"size": 13, "bold": True, "color": FAIL}),
        ("Finish time is still the sum of rooms. Interval is not the result.", {"size": 12}),
    ])
    y = Inches(2.7)
    for lab, cyc, note in [
        ("load", "~4100", "DRAM A/B"),
        ("compute", "~4816", "16×4, II=1"),
        ("store", "~4100", "DRAM C"),
    ]:
        labeled_box(s, Inches(7.15), y, Inches(5.4), Inches(0.7), f"{lab}   {cyc}   ·  {note}", fill=WHITE, line=NAVY, size=14)
        if lab != "store":
            add_textbox(s, Inches(9.5), y + Inches(0.62), Inches(1.2), Inches(0.22), "↓  not hidden", size=10, color=FAIL, align=PP_ALIGN.CENTER)
        y += Inches(0.92)
    add_textbox(s, Inches(6.95), Inches(5.55), Inches(5.8), Inches(1.2),
                "Aug 23 overlap seed 8678 / 320: architecture gate fail (coarse tile overlap, inner stretches restart). Manual pe_pp 9640 / 352; pe_pp_plus 4688 / 352 — still not rank-1 I/O. mmflow overlap re-run Slurm 2789637: LLM timed out.",
                size=12, color=INK)
    notes(s, "Stream is packed FIFOs + Crow ram_2p + flatten pe_kj, not the existence of a DATAFLOW pragma. Rank-1 interval ≈ latency because of K-tiles / B ping-pong / C drain.")


def s_stream_not_pp(prs, n, total):
    s = blank(prs)
    header(s, "4. Stream is not DATAFLOW + ping-pong on multi-PE", "Comparison 1  ·  finish time, not interval")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Design", "Latency", "DSP", "What actually overlapped"],
            ["Enforcement (column 3)", "12 893", "320", "Pragma shape. Finish time ≠ max(modules)."],
            ["Aug 23 overlap seed (not mmflow)", "8678", "320", "Coarse tile overlap. Architecture gate fail."],
            ["Manual pe_pp on mmflow multi-PE", "9640", "352", "csim pass. Still not rank-1 I/O."],
            ["Manual pe_pp_plus", "4688", "352", "csim pass. Still not the stream pack."],
            ["Stream hide load/store", "4292 / 4216", "320", "Load, 16 PEs, store together. Packed ap_uint<128>, Crow ram_2p, flatten pe_kj 4096."],
        ],
        [Inches(3.3), Inches(1.6), Inches(0.9), Inches(6.6)],
        font=13,
    )
    add_textbox(s, ML, Inches(5.55), Inches(12.4), Inches(1.4),
                "Rank-1 interval ≈ latency because of a systolic I/O schedule (K-tiles, B ping-pong, C drain), not because a DATAFLOW pragma exists.\n"
                "Do not quote enforcement interval 4553 as the result.",
                size=14, color=NAVY, bold=True)


def s_wave1(prs, n, total):
    s = blank(prs)
    header(s, "5. Other AutoSA mm-family benches, ≤ 2% of rank-1", "Gate: agent latency ≤ rank-1 × 1.02  ·  campaign 20260825_wave1_aav_n_gf")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Bench", "Agent", "Rank-1", "DSP", "vs ×1.02"],
            ["autosa_mm (mmflow / noskills)", "4292 / 4216", "4228", "320", "pass"],
            ["autosa_mm_hcl", "4294", "4230", "320", "pass 1.51%"],
            ["autosa_mm_hcl_intel", "4292", "4226", "320", "pass 1.56%"],
            ["autosa_mm_int16", "4280", "4219", "64", "pass 1.45%"],
            ["autosa_mm_catapult", "8351", "8286", "96", "pass 0.78%"],
            ["autosa_mm_intel", "4525", "4178", "640", "fail 8.3% after retry 2789605 (ceiling 4261)"],
            ["autosa_mm_getting_started", "4525", "2194", "640", "fail ~2× (rank-1 is two DATAFLOW tiles; agent I/PE=2)"],
        ],
        [Inches(3.3), Inches(1.6), Inches(1.3), Inches(0.9), Inches(5.3)],
        font=13,
    )
    add_textbox(s, ML, Inches(6.15), Inches(12.4), Inches(0.85),
                "Four of six wave1 benches pass. Intel compute-rewrite LLM timed out; stream still ran and promoted 4525 / 640. Not this round: cnn, lu, large_*, HBM.",
                size=14, color=INK)


def s_why_320(prs, n, total):
    s = blank(prs)
    header(s, "6. Why 320 DSP", "U280 has 9024 DSP. Rank-1 320 is iso-compute, not a fill of the chip.")
    footer(s, n, total)
    tb(s, ML, Inches(0.95), Inches(12.4), Inches(0.7), [
        ("320  =  16 PE × SIMD 4 × ~5 DSP per float MAC at II=1. AutoSA chose that array for 64³ autosa_mm. The locked slide holds that point.", {"size": 14}),
    ])
    add_table(
        s, ML, Inches(1.7), Inches(12.4),
        [
            ["Array", "Stage", "Latency", "DSP", "vs 4228 / 320"],
            ["16×4 (locked)", "stream", "4292", "320", "+1.5%"],
            ["16×4", "stream, empty JSON", "4216", "320", "−0.3%"],
            ["32×8  20260830_mm32x8", "compute rewrite", "9782", "1344", "more silicon, still sequential I/O"],
            ["32×8", "stream", "4583", "1280", "worse than 16×4 stream (I/PE=2, pe_kj=512 × two tiles)"],
            ["Family C iso-DSP 32-PE", "io4_8x4_s8_k32_j64", "2245", "1280", "misses AutoSA cand 9 ≤1846 at ~1296 DSP"],
            ["Family C selected", "io4_16x8_s8_k32_j64", "1571", "5120", "faster only by oversubscribing DSP"],
        ],
        [Inches(2.7), Inches(2.5), Inches(1.3), Inches(1.1), Inches(4.8)],
        font=12,
    )
    add_textbox(s, ML, Inches(6.15), Inches(12.4), Inches(0.85),
                "AutoSA exhaustive candidate 9 (2-D, not Design 1) is 1846–2161 / ~1296 DSP. Candidate 5 is 2033–2656. Paper Design 5 ≈ cand 9. Efficiency vs cand 9 is not closed. P&R / bitstream later.",
                size=14, color=INK)


def s_pack_generic(prs, n, total):
    s = blank(prs)
    header(s, "Skill pack 1. Generic HLS",
           "Transfer test: kernel-independent INTERFACE / PIPELINE / UNROLL / PARTITION / burst / LCST  ·  comparison 1 flash")
    footer(s, n, total)
    add_textbox(s, ML, Inches(0.88), Inches(12.4), Inches(0.4),
                "skills_ii_target_miss_solutions_added(90skills).json  (123 entries)  +  overlay flash_no_RMW_m_axi_skill_entries.json (33). Tagged kernel-independent.",
                size=13, color=MUTED)
    add_table(
        s, ML, Inches(1.28), Inches(12.4),
        [
            ["Family", "Exemplar skill id", "One-line role"],
            ["Burst / INTERFACE", "axi-burst-widening-512, axi-burst-coalescing-narrow-safe", "512-bit m_axi widening; keep scalar pointer ABI"],
            ["PIPELINE", "prompt-pipeline, hls-pipeline-hot-loop-achieve-ii", "Pick the hot loop; resolve II; not pragma-only"],
            ["UNROLL", "prompt-unroll, hls-unroll-independent-loop", "Unroll independent loops after independence is proven"],
            ["PARTITION", "partition-cyclic-on-port-conflict", "Bank local arrays to match unroll / pipeline ports"],
            ["LCST staging", "hls-pipeline-stage-global-memory, local-axi-staging-for-ii", "Stage off m_axi before compute"],
            ["No RMW overlay", "avoid-no-rmw-m_axi-direct, hls-load-compute-store-no-rmw-m_axi", "No m_axi accumulate; distinct gmem bundles"],
        ],
        [Inches(2.15), Inches(4.55), Inches(5.7)],
        font=12,
    )
    labeled_box(s, ML, Inches(5.15), Inches(12.4), Inches(0.95),
                "Contamination (transfer test fails): hls-baseline-load-compute-store-gate and hls-doublebuffer-load-compute-store\n"
                "encode PE_BLK ∈ {16,32,64}, 64×64 float LANES=16, and GEMM ping-pong buf[2]. That is not kernel-independent.\n"
                "This kernel: mmflow flash 139 484 cycles / 10 DSP. Legal. No 16×4 array. Not a systolic pass.",
                fill=PALE, line=GOLD, size=13, tcolor=INK, align=PP_ALIGN.LEFT)
    notes(s, "Classify by transfer, not filename. Mixing this pack into flash does not invent PEs.")


def s_pack_gemm(prs, n, total):
    s = blank(prs)
    header(s, "Skill pack 2. GEMM-family  (not AutoSA, not systolic I/O)",
           "gemm_flatten_v1  +  post_flash_pe_skill_entries.json  +  post_flash_pe_recipe.py  ·  comparison 1 compute rewrite")
    footer(s, n, total)
    add_textbox(s, ML, Inches(0.88), Inches(12.4), Inches(0.32),
                "File on disk is still named post_flash_dse_pe_skill_entries.json. Slides: compute rewrite. IDs below are the file ids.",
                size=12, italic=True, color=MUTED)
    add_table(
        s, ML, Inches(1.22), Inches(12.4),
        [
            ["Skill id", "Role"],
            ["hls-gemm-static-nest-flatten-and-k-partition", "Flatten i/k/j on a static nest; partition K so unrolled MACs have ports"],
            ["hls-avoid-serial-fp-acc-under-full-k-unroll", "Reject a single FP acc after claiming full-K unroll"],
            ["hls-avoid-pipeline-innermost-only-goal-on-gemm", "Innermost-only PIPELINE is not the GEMM pass"],
            ["hls-avoid-outer-sequential-gemm-after-inner-pipeline", "Do not leave i sequential after an inner II=1"],
            ["hls-dse-gemm-multi-pe-latency-hiding", "Rewrite compute nest into row PEs; pipeline independent j; keep LCST ABI"],
            ["hls-dse-gemm-simd-k-adder-tree", "SIMD-k=4 unrolled adder tree; pipelined loop is j, not k"],
            ["hls-dse-partition-match-pe-simd", "Crow complete on PE dim; A cyclic PE×SIMD; B cyclic SIMD on K"],
            ["avoid-dse-k-recurrence-on-single-c", "Reject k-loop recurrence on one C element (II=4, DSP~3–8)"],
            ["hls-dse-pick-pe-simd-for-kernel", "Read kernel.h + recipe. autosa_mm default 16×4; not a search"],
        ],
        [Inches(4.7), Inches(7.7)],
        font=12,
    )
    add_textbox(s, ML, Inches(6.35), Inches(12.4), Inches(0.65),
                "This kernel: compute rewrite 13 160 / 352 DSP. Compute nest ~4816 at II=1. Remaining latency is sequential load ~4100 + store ~4100. Recipe for autosa_mm is PE=16 SIMD=4 (post_flash_pe_recipe.py), not a design-space search.",
                size=13, color=NAVY, bold=True)
    notes(s, "Never say DSE. The PE recipe is fixed for the locked slide.")


def s_pack_stream(prs, n, total):
    s = blank(prs)
    header(s, "Skill pack 3. Systolic / hide load-store",
           "post_flash_stream_pe_io_skill_entries.json  (6 skills)  ·  AutoSA I/O plan, not kernel0  ·  comparison 1")
    footer(s, n, total)
    add_table(
        s, ML, Inches(0.95), Inches(12.4),
        [
            ["Skill id", "Role"],
            ["hls-stream-pe-array-dataflow", "PE_NUM mm_pe tasks under DATAFLOW; packed ap_uint FIFOs; Crow ram_2p; flatten pe_kj; fuse init/drain"],
            ["hls-stream-io-overlap-dram", "load_A / load_B / store_C as INLINE-off DRAM tasks; compute never touches m_axi"],
            ["hls-stream-b-systolic-forward", "Packed vec4_bits along the PE chain; always forward; drain_B consumes fifo_B[PE_NUM]"],
            ["hls-stream-match-pe-pack-to-data-t", "Pack SIMD as ap_uint<pack_bits>; float 16×4 → ap_uint<128>"],
            ["hls-stream-tile-loop-around-dataflow", "If I/PE>1, wrap DATAFLOW in the top i-tile; do not loop tiles inside mm_pe"],
            ["avoid-stream-shared-local-abc", "Reject shared on-chip A/B/C across DATAFLOW tasks; reject kernel0 clone"],
        ],
        [Inches(4.15), Inches(8.25)],
        font=13,
    )
    add_textbox(s, ML, Inches(5.55), Inches(12.4), Inches(1.4),
                "autosa_mm: PE_NUM=16, SIMD=4, pe_kj = (K/SIMD)×J = 1024 beats per I-strip, 4096 across four I-strips. Crow BIND_STORAGE ram_2p — never complete-partition J.\n"
                "This kernel: stream 4292 / 320 (90-skill mmflow) or 4216 / 320 (empty JSON, same PE recipe). Failures taught here: struct-of-float FIFOs 16 878 / 80; Crow complete-partition J → II=4; flatten-off 7162.",
                size=14, color=NAVY)
    notes(s, "This pack is hide load-store on the 16×4 array. Mixing it into flash does not invent PEs.")


def s_pack_overlap(prs, n, total):
    s = blank(prs)
    header(s, "Skill pack 4. Overlap / ping-pong PE",
           "post_flash_overlap_pe_skill_entries.json  (3 skills)  ·  C2HLS_ENFORCEMENT  ·  comparison 1  ·  not rank-1")
    footer(s, n, total)
    add_table(
        s, ML, Inches(0.95), Inches(12.4),
        [
            ["Skill id", "Role"],
            ["hls-overlap-fuse-crow-init-drain", "Delete init_pe_j / wb_pe_j. First k-group writes; last k-group writes C. Same 16×4 MAC."],
            ["hls-overlap-pingpong-dataflow-on-pe-array", "INLINE-off load_tile / compute_tile / store_tile. Outer i-tile wraps DATAFLOW. buf[2] A/C. Keep one compute_tile."],
            ["avoid-overlap-stream-pe-fifo-pack", "Reject mm_pe + packed FIFO chain in this ablation. That is pack 3, already measured."],
        ],
        [Inches(4.55), Inches(7.85)],
        font=13,
    )
    labeled_box(s, ML, Inches(4.15), Inches(12.4), Inches(2.7),
                "Judge: ping-pong DATAFLOW present on code and csynth (20260829_123043).\n"
                "Finish time 12 893 / 320 DSP — still load + compute + store. Overlap gate fails.\n"
                "Aug 23 overlap seed 8678 / 320: architecture gate fail (coarse tiles, inner stretches restart).\n"
                "A DATAFLOW pragma is not hide load-store. Rank-1 interval ≈ latency because of K-tiles, B ping-pong, and C drain.",
                fill=PALE, line=FAIL, size=14, tcolor=INK, align=PP_ALIGN.LEFT)
    notes(s, "Do not quote enforcement interval as the result. This column is not the systolic pass.")


def s_pack_onchip(prs, n, total):
    s = blank(prs)
    header(s, "Skill pack 5. On-chip spend-DSP",
           "flash_onchip_wide_gemm_skill_entries.json  ·  PACKAGED_SKILLS_ONLY=1  ·  comparison 2  ·  not mixed into the 90-skill dump")
    footer(s, n, total)
    add_table(
        s, ML, Inches(0.92), Inches(12.4),
        [
            ["Skill id", "Role"],
            ["hls-onchip-stage-when-fits", "Stage all of A and B; compute into C_local; never touch m_axi in compute. 64³ float fits."],
            ["hls-onchip-axi-512-three-bundles", "gmem0/1/2, max_widen_bitwidth=512, LANES=16. Load/store trip ≈ 256, not 4096"],
            ["hls-onchip-write-once-affine-c", "Nested affine i / j+=PE_BLK. C_local[row][j0+p]=dot. No g>>k, no C RMW"],
            ["hls-onchip-flatten-c-output-groups", "One II=1 pipeline over I×(J/PE_BLK) groups. PE_BLK 16/32/64 after II=1"],
            ["hls-onchip-full-k-unroll-adder-tree", "Full UNROLL k into partial sums then a tree. Not a serial FP acc"],
            ["hls-onchip-partition-match-unroll", "A_local complete on K; B_local enough for PE_BLK; C_local complete on J group"],
            ["avoid-onchip-dataflow-when-io-matches-compute", "Reject DATAFLOW/ping-pong when compute ≈ 512-bit load window. 940 = max(A,B)+compute+store"],
            ["avoid-onchip-kernel0-systolic", "Keep autosa_mm ABI. Do not emit kernel0 / L2 FIFOs / mm_pe chain"],
        ],
        [Inches(4.55), Inches(7.85)],
        font=11,
    )
    add_textbox(s, ML, Inches(5.85), Inches(12.4), Inches(1.1),
                "Champion class: 940 / 5344, Cosim PASS 1071. Nested i / j+=LANES; two independent load modules.\n"
                "Fusion of A+B in one linearized-t loop is a failure mode (871 csynth / 1224 cosim) — not the champion, not comparison 1.\n"
                "JSON on disk also has later entries (avoid-onchip-fused-ab-lockstep, avoid-onchip-dsp-over-u280, row-uf, K-tile under DSP cap). Distilled 8 above are the 940-class pack.",
                size=13, color=NAVY)
    notes(s, "Never put 940 next to 4228. Keep this pack out of the 90-skill dump and frozen mmflow.")


def s_spend_ladder(prs, n, total):
    s = blank(prs)
    header(s, "8. Flash must stop under-using DSP  —  one change per step", "Comparison 2  ·  flash-only  ·  not a fair beat of 4228")
    footer(s, n, total)
    data = CategoryChartData()
    data.categories = ["min DSP 500", "+512-bit I/O", "+write-once C", "+PE_BLK=16"]
    data.add_series("Csynth latency (cycles)", [9423, 2760, 1261, 940])
    chart = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, ML, Inches(0.95), Inches(12.4), Inches(3.55), data).chart
    chart.has_legend = False
    try:
        chart.plots[0].has_data_labels = True
        chart.plots[0].data_labels.font.size = Pt(11)
    except Exception:
        pass
    add_table(
        s, ML, Inches(4.55), Inches(12.4),
        [
            ["Step (one change)", "Campaign", "Latency", "DSP", "Then the report said"],
            ["DSP floor 500", "…_dsp500_20260904_085931", "9423", "1344", "More MACs. Load/store still ~4k. LCST."],
            ["512-bit A/B/C, LANES=16, gmem0/1/2", "…_wideio_20260904_100437", "2760", "636", "I/O ~258. Compute 2099, Final II=4 (g>>k + C RMW)."],
            ["Affine i, j+=PE; C_local=dot", "…_computeii_20260904_120524", "1261", "2672", "Compute II=1."],
            ["PE_BLK=16", "…_pe16_20260904_131622", "940", "5344", "Two load modules. Type no. Cosim PASS 1071."],
        ],
        [Inches(3.15), Inches(2.85), Inches(1.05), Inches(0.9), Inches(4.45)],
        font=11,
    )
    add_textbox(s, ML, Inches(6.85), Inches(12.4), Inches(0.35),
                "Ladder ends at PE_BLK=16. j-tile ping-pong and LLM tile-pp were worse (not shown).",
                size=12, italic=True, color=MUTED)


def s_tiling_spend(prs, n, total):
    """Figure C: spend-DSP champion vs fused load. Comparison 2 only."""
    s = blank(prs)
    header(s, "Tiling C. Spend-DSP champion is not systolic tiles",
           "Comparison 2  ·  PE_BLK=16  ·  5344 DSP  ·  do not place beside 4228")
    footer(s, n, total)

    rect(s, ML, Inches(0.95), Inches(6.15), Inches(6.05), PALE, RULE)
    labeled_box(s, Inches(0.55), Inches(1.05), Inches(5.95), Inches(0.4),
                "Champion  ·  nested i / j += LANES  ·  940 csynth / 1071 cosim", fill=NAVY, line=NAVY, size=13, tcolor=WHITE)
    tb(s, Inches(0.65), Inches(1.55), Inches(5.85), Inches(1.15), [
        ("Not Design 1. No K-tiles. No B ping-pong. No mm_pe FIFOs. Full A, B, C staged on-chip (64³ float fits).", {"size": 13}),
        ("Nested load: for i … for j += LANES=16. Two independent load functions (separate stall domains).", {"size": 13}),
    ])
    # LCST diagram
    labeled_box(s, Inches(0.75), Inches(2.85), Inches(2.55), Inches(1.15), "load_A\n259 cycles\nFF 540", fill=TILE, line=NAVY, size=13)
    labeled_box(s, Inches(3.55), Inches(2.85), Inches(2.55), Inches(1.15), "load_B\n258 cycles\nstarted together", fill=TILE, line=NAVY, size=13)
    add_textbox(s, Inches(0.75), Inches(4.05), Inches(5.35), Inches(0.25), "max(259, 258)  then", size=12, color=MUTED, align=PP_ALIGN.CENTER)
    labeled_box(s, Inches(0.75), Inches(4.3), Inches(5.35), Inches(0.7), "compute_blocks   340   ·  PE_BLK=16  ·  full-K adder tree  ·  5344 DSP", fill=WHITE, line=NAVY, size=12)
    labeled_box(s, Inches(0.75), Inches(5.1), Inches(5.35), Inches(0.55), "store_C   260   ·  WREADY in-loop; BVALID at parent", fill=WHITE, line=NAVY, size=12)
    add_textbox(s, Inches(0.65), Inches(5.9), Inches(5.85), Inches(0.85),
                "940 ≈ max(259, 258) + 340 + 260. Pipeline type: no. AXI 512-bit, three bundles, latency=32.",
                size=13, bold=True, color=NAVY)

    rect(s, Inches(6.8), Inches(0.95), Inches(6.1), Inches(6.05), PALE, RULE)
    labeled_box(s, Inches(6.9), Inches(1.05), Inches(5.9), Inches(0.4),
                "Failure mode  ·  fused linearized-t load  ·  not the champion", fill=FAIL, line=FAIL, size=13, tcolor=WHITE)
    tb(s, Inches(6.95), Inches(1.55), Inches(5.8), Inches(1.5), [
        ("8-skill 7-turn 20260906_104733: 871 csynth / 5088 DSP, Cosim PASS 1224.", {"size": 13, "bold": True, "color": FAIL}),
        ("A and B in one II=1 loop over linearized t. Csynth priced lockstep II=1; RTL stalls if either gmem0 or gmem1 RVALID/ARREADY is 0.", {"size": 12}),
    ])
    labeled_box(s, Inches(7.05), Inches(3.2), Inches(5.6), Inches(1.15), "one fused load_ab loop\nlinearized t\n~132 k FF in load vs champion load_A 540 FF", fill=WHITE, line=FAIL, size=13, tcolor=FAIL)
    labeled_box(s, Inches(7.05), Inches(4.45), Inches(5.6), Inches(0.85), "store waits BVALID inside the pipeline\nCosim 1224  >  champion Cosim 1071", fill=WHITE, line=FAIL, size=12)
    add_textbox(s, Inches(6.95), Inches(5.45), Inches(5.8), Inches(1.3),
                "Repro 20260906_123340: 1002 / 5344, split load_a/load_b, still linearized t — not nested i / j+=LANES. Fusion is model invention, not instructed. Do not put 871 on the 4228 slide.",
                size=12, color=INK)
    notes(s, "j-tile ping-pong was worse (not shown). Champion remains 940. DSP 5344 = 59% of 9024, 177% of one SLR — allowed on comparison 2.")


def s_module_940(prs, n, total):
    """First-class 940 module table from csynth.rpt."""
    s = blank(prs)
    header(s, "Latency per module  —  PE_BLK=16 champion (csynth)",
           "Report: autosa_mm_csynth.rpt  ·  4 Sep 2026 15:33  ·  Vitis HLS 2023.2  ·  comparison 2")
    footer(s, n, total)

    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Instance (csynth)", "Module", "Latency min–max", "Pipeline type", "DSP", "FF"],
            ["grp_…_load_A_rows_load_A_cols_fu_17020", "Pipeline_load_A_rows_load_A_cols", "259–259", "no", "0", "540"],
            ["grp_…_load_B_rows_load_B_cols_fu_17091", "Pipeline_load_B_rows_load_B_cols", "258–258", "no", "0", "131 110"],
            ["grp_…_compute_blocks_fu_21194", "Pipeline_compute_blocks", "340–340", "no", "5344", "615 512"],
            ["grp_…_store_C_rows_store_C_cols_fu_25422", "Pipeline_store_C_rows_store_C_cols", "260–260", "no", "0", "2668"],
            ["autosa_mm (kernel)", "—", "940–940", "no", "5344", "765 008"],
        ],
        [Inches(3.55), Inches(3.35), Inches(1.7), Inches(1.35), Inches(1.05), Inches(1.4)],
        font=12,
    )
    labeled_box(s, ML, Inches(4.55), Inches(12.4), Inches(0.7),
                "940  ≈  max(259, 258)  +  340  +  260     ·     load A and load B are two modules, started together",
                fill=NAVY, line=NAVY, size=16, tcolor=WHITE)
    tb(s, ML, Inches(5.4), Inches(12.4), Inches(1.55), [
        ("Total resources (same report): BRAM_18K 90  ·  DSP 5344  ·  FF 765 008  ·  LUT 346 116  ·  URAM 0. Available device DSP 9024 (59%). Available SLR DSP 3008 (177% of one SLR) — allowed because comparison 2 asked for device DSP, not iso-320.", {"size": 14}),
        ("Cosim: artifacts/pc2/manual_pe16_tile_pp/cosim_pe16.json  ·  PASS 1071 cycles, gold match. Target clock 3.33 ns; estimated clock is not a comparison metric.", {"size": 13, "color": MUTED}),
    ])
    notes(s, "Quote latency min/max 940 and DSP 5344. Do not quote interval 941 or estimated 2.431 ns as the result.")


def s_module_across(prs, n, total):
    s = blank(prs)
    header(s, "Module-level latency across designs", "Csynth unless Cosim is named  ·  keep the two comparisons in separate columns")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Design", "Comparison", "Modules (cycles)", "One-call finish"],
            ["Compute rewrite 13 160 / 352", "1  (iso-320)", "compute ~4816 II=1; load ~4100; store ~4100 (serial LCST)", "sum: load + compute + store"],
            ["Enforcement 12 893 / 320", "1", "DATAFLOW pragma present; rooms still sequential", "still load+compute+store; overlap gate fail"],
            ["Stream 4292 / 4216 / 320", "1", "load, 16 PEs, store overlapped; flatten pe_kj 4096", "≈ slowest room (hide load-store)"],
            ["AutoSA rank-1 4228 / 320", "1", "K-tiles, B ping-pong, C drain; interval ≈ latency", "systolic I/O schedule, not a pragma"],
            ["PE_BLK=16  940 / 5344", "2  (spend DSP)", "load_A 259, load_B 258, compute 340, store 260", "max(A,B)+compute+store; Cosim 1071"],
            ["Fused 8-skill 871 csynth / 5088", "2  (not champion)", "fused A+B lockstep; store BVALID in-loop", "Cosim PASS 1224 — not 871"],
        ],
        [Inches(3.15), Inches(1.55), Inches(4.35), Inches(3.35)],
        font=12,
    )
    add_textbox(s, ML, Inches(6.25), Inches(12.4), Inches(0.75),
                "Do not call 871 the champion. Csynth priced fused II=1; RTL did not. Independent load modules (259 ∥ 258) are the 940 I/O shape.",
                size=14, bold=True, color=NAVY)


def s_eight_skill(prs, n, total):
    s = blank(prs)
    header(s, "Distilled 8-skill pack  —  a separate question", "Not step 5 of the spend-DSP ladder  ·  more than one code choice moved")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Run", "Csynth", "Cosim", "I/O shape"],
            ["8-skill, 7-turn, 20260906_104733", "871 / 5088 DSP", "PASS 1224", "Fused load_ab, linearized t. RTL stalls if either RVALID is 0."],
            ["8-skill, 7-turn, 20260906_123340", "1002 / 5344 DSP", "not run", "Split load_a / load_b, still linearized t, not nested i, j+=LANES."],
            ["PE=16 champion (ladder endpoint)", "940 / 5344 DSP", "PASS 1071", "Nested i / j+=LANES. Two load modules. Store WREADY only."],
        ],
        [Inches(3.5), Inches(1.9), Inches(1.6), Inches(5.4)],
        font=13,
    )
    tb(s, ML, Inches(4.15), Inches(12.4), Inches(2.7), [
        ("Pack: flash_onchip_wide_gemm_skill_entries.json, PACKAGED_SKILLS_ONLY=1, judges MIN_DSP=500 and PE_BLK=16. Not mixed into the 90-skill dump or frozen mmflow.", {"size": 14}),
        ("Why the LLM fused A and B: the prompt says latency = max(load_A,load_B)+compute+store and “overlapping A/B loads is enough,” and item 6’s trip target is one ~256-cycle load loop. Ignore FLASH item 5 kills DATAFLOW ping-pong, not dual-issue in one loop. Fusion is a reading of overlap, not a hard forbid being ignored.", {"size": 14}),
    ])


def s_avoid(prs, n, total):
    s = blank(prs)
    header(s, "9. Stream failures that became avoid-rules", "Iso-compute path  ·  DSP as II proxy on this kernel")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Attempt", "Cycles", "DSP", "Failure"],
            ["Struct-of-floats FIFOs", "16 878", "80", "Four serialized float reads, II=4"],
            ["Packed ap_uint<128> only", "18 607", "80", "Still II=4"],
            ["Crow complete-partition on J", "same class", "80", "mux_case_0 FP recurrence, II=4"],
            ["Crow ram_2p, flatten-off pe_k0", "7162", "320", "16 k-tiles × 4 i-tiles refill of depth ~35"],
            ["Flatten pe_kj 1024/4096, fuse init/drain", "4285 then 4292", "320", "Aug 18 then frozen mmflow"],
        ],
        [Inches(4.1), Inches(1.9), Inches(1.1), Inches(5.3)],
        font=13,
    )
    add_textbox(s, ML, Inches(5.55), Inches(12.4), Inches(1.3),
                "DSP as II proxy on this kernel: 80 DSP ≈ 16 PEs × ~5 at II=4.  320 DSP ≈ 16 × 20 at II=1.",
                size=16, bold=True, color=NAVY)


def s_open(prs, n, total):
    s = blank(prs)
    header(s, "10. Limits and open items", "As requested, not as a hedge")
    footer(s, n, total)
    add_table(
        s, ML, Inches(1.0), Inches(12.4),
        [
            ["Item", "Status"],
            ["Csynth + csim", "Done on all quoted iso-compute numbers"],
            ["Cosim", "Off for mmflow / wave1 / ablation. On for PE=16 champion (1071 PASS) and fused 8-skill kernel (1224 PASS)"],
            ["Bitstream / P&R / extra HBM", "Not done"],
            ["One model", "DeepSeek-v4-flash"],
            ["Clone AutoSA compiler / emit kernel0 as agent result", "Not the method"],
            ["Iso-compute vs rank-1 Design 1", "Closed: 4216–4292 vs 4228 at 320 DSP"],
            ["Wave1 2%", "4/6 pass; intel and getting_started fail after retry"],
            ["Efficiency vs cand 9 (1846 / ~1296 DSP)", "Not closed. Needs K-tile + L1/L2 I/O, not more PEs"],
            ["Frozen artifacts", "Do not overwrite 20260830_mmflow, 32×8, family C first campaign, Aug 18 / Sep 2 briefs"],
        ],
        [Inches(4.5), Inches(7.9)],
        font=13,
    )


def s_takeaways(prs, n, total):
    s = blank(prs)
    header(s, "What to take from the study", "One kernel  ·  two comparisons  ·  one variation per step on spend-DSP")
    footer(s, n, total)
    bullets = [
        ("Iso-compute is closed at Design 1 class.", "4216–4292 vs 4228 at 320 DSP. Empty skills JSON still closes if the PE recipe + stream skeleton remain."),
        ("Generic HLS skills are not the systolic pass.", "90-skill flash 139 484 / 10 DSP is a legal kernel with almost no MAC parallelism. Enforcement DATAFLOW+ping-pong is 12 893, not rank-1 I/O."),
        ("Stream ≠ a DATAFLOW pragma.", "Packed ap_uint<128>, Crow ram_2p, flatten pe_kj 4096, A private, B forward, C drain."),
        ("Rank-1 tiles are 16×32 on I×K, J not coarse-tiled.", "array_part[16,64,32], 4 I-tiles, 2 K-tiles, B 64×32 ping-pong, C drain on last K-tile. Frozen agent stream parked full B; still 4292."),
        ("Spend-DSP is a different experiment.", "940 ≈ max(259,258)+340+260 at 5344 DSP. Cosim 1071. Not a fair beat of 4228. 871 csynth is not the RTL finish time."),
        ("Cand-9 efficiency is not closed.", "1846–2161 / ~1296 DSP needs K-tile + L1/L2 I/O, not more PEs."),
    ]
    y = Inches(0.98)
    for i, (h, b) in enumerate(bullets, 1):
        labeled_box(s, ML, y, Inches(0.42), Inches(0.42), str(i), fill=NAVY, line=NAVY, size=16, tcolor=WHITE)
        add_textbox(s, Inches(1.05), y, Inches(11.8), Inches(0.24), h, size=16, bold=True, color=NAVY)
        add_textbox(s, Inches(1.05), y + Inches(0.26), Inches(11.8), Inches(0.55), b, size=13, color=INK)
        y += Inches(0.95)


def main():
    prs = new_prs()
    builders = [
        s_title,
        s_protocol,
        s_two_comparisons,
        s_method,
        s_pipeline_chart,
        s_stage_table,
        s_ablation,
        s_flash_weak,
        s_tiling_rank1,
        s_tiling_stream,
        s_stream_not_pp,
        s_wave1,
        s_why_320,
        s_pack_generic,
        s_pack_gemm,
        s_pack_stream,
        s_pack_overlap,
        s_pack_onchip,
        s_spend_ladder,
        s_tiling_spend,
        s_module_940,
        s_module_across,
        s_eight_skill,
        s_avoid,
        s_open,
        s_takeaways,
    ]
    total = len(builders)
    for i, fn in enumerate(builders, 1):
        fn(prs, i, total)
    prs.save(str(OUT))
    print(f"Wrote {OUT}  ({total} slides)")
    return OUT, total


if __name__ == "__main__":
    main()
