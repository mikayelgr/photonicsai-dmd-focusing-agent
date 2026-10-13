"""Builds the Task 2 report PDF from results/experiments.jsonl."""
import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

RESULTS = os.environ.get("FOCUS_RESULTS_DIR", "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else "Task2_report.pdf"
GITHUB = os.environ.get("GITHUB_URL", "https://github.com/&lt;user&gt;/&lt;repo&gt; (fill in)")

rows = [json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))]
best = max(rows, key=lambda r: r["enhancement"])

ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, keepWithNext=1)
cap = ParagraphStyle("c", parent=body, fontSize=8.5, leading=11, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"))
cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
code = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#f3f3f3"), borderPadding=4)

s = []
P = lambda t, st=body: s.append(Paragraph(t, st))

P("Homework 2, Task 2: Focusing through a simulated bench", h1)
P("PhotonicsAI | AI-Driven Science and Autonomous Laboratories", cap)
s.append(Spacer(1, 8))
P(f"<b>Code:</b> {GITHUB} (folder <font face='Courier'>task2/</font>). "
  "<b>Screen recording:</b> <font face='Courier'>task2/agent_focus_run.mp4</font> in the same repository.")

P("1. Simulated bench", h2)
P("The simulation covers the image-forming part of the path: laser, DMD, scattering medium (RSM), camera. "
  "The DMD has 32 x 32 mirrors and modulates amplitude only: each mirror is on (1) or off (0). "
  "The camera has 64 x 64 pixels and records")
P("I = |T E<sub>in</sub>|<super>2</super> + noise", ParagraphStyle("eq", parent=body, alignment=TA_CENTER))
P("T is a 4096 x 1024 complex matrix. Each column is the speckle field one mirror produces on the camera: "
  "circular complex Gaussian noise filtered by a circular pupil in k-space, so speckle grains are about 2 pixels "
  "across, as behind a real diffuser and objective. T is scaled so the all-on pattern gives 2000 photons per pixel "
  "on average. Each shot adds Poisson shot noise and Gaussian read noise (sigma = 2 counts), so two shots of the "
  "same pattern differ. The medium (T) is fixed for the whole session, like a sample sitting on the bench.")

P("2. Focusing algorithm (the tool)", h2)
P("<font face='Courier'>focus_beam(macropixel_size, focus_size, bg_inner, bg_outer, n_bg_regions, iterations, seed, reason)</font> "
  "follows the algorithm from the assignment:")
for t in [
    "The DMD is divided into macropixels of macropixel_size x macropixel_size mirrors and starts from a random on/off pattern.",
    "After each transmission (one camera shot) the tool reads the mean intensity in a square focus region at the camera centre "
    "and in a proximal background annulus (radii bg_inner to bg_outer).",
    "The annulus is split into n_bg_regions angular sectors. Enhancement = I<sub>focus</sub> / mean of the sector means, "
    "so one bright speckle in the ring cannot dominate the background estimate.",
    "Each iteration flips k random macropixels, measures again and keeps the flip if the enhancement rose. k shrinks linearly "
    "from N/8 to 1, so early steps explore and late steps fine-tune. Every 25 iterations the kept pattern is re-measured, "
    "so a single noisy high shot cannot lock in.",
    "The reported enhancement is the mean over 5 fresh shots of the final pattern. The tool also reports the classic "
    "reference (optimised focus intensity over focus intensity averaged across 20 random patterns) and the binary-amplitude "
    "theory N/(2 pi) + 1/2.",
]:
    P("&bull; " + t)
P("Each call saves the initial and final camera images, the DMD pattern and the enhancement curve under "
  "<font face='Courier'>results/run_XX/</font>, and appends one row to <font face='Courier'>results/experiments.jsonl</font>. "
  "The table and figures below are generated from that log.")

P("3. Agent integration", h2)
P("The tool is added to the Homework 1, Task 6 graph as a LangChain tool of the executor, next to its coding tool "
  "(<font face='Courier'>agent_integration.py</font>). The orchestrator still routes by a model call. When the user asks "
  "to focus the beam, it routes to the executor. The executor's system prompt describes the bench (DMD size, camera size, "
  "speckle grain) and tells it to plan a short campaign on its own: a coarse check first, then smaller macropixels with "
  "iterations scaled to the number of macropixels, at least one change of region sizes, 4-6 runs in total, with a "
  "one-sentence reason per call. It never asks the user for settings. After each tool result the executor keeps the best "
  "enhancement in the graph state (<font face='Courier'>best_enhancement</font>), and its final message reports all runs. "
  "The user types only the initial request:")
P("user: Focus the beam on the bench.", code)

s.append(PageBreak())
P("4. Experiments run", h2)
hdr = ["Run", "Macro-\npixel", "N", "Focus\n(px)", "BG ring\n(px)", "BG\nsectors", "Iter.",
       "Initial\neta", "Final\neta", "eta vs\nrandom", "Theory\nN/2pi", "Reason"]
data = [hdr]
for r in rows:
    st = r["settings"]
    data.append([r["run_id"], st["macropixel_size"], st["n_macropixels"], f'{st["focus_size"]}x{st["focus_size"]}',
                 f'{st["bg_inner"]}-{st["bg_outer"]}', st["n_bg_regions"], st["iterations"],
                 r["initial_enhancement"], f'<b>{r["enhancement"]}</b>', r["enhancement_vs_random"],
                 r["theory_binary_amplitude"], Paragraph(r["reason"], cell)])
hcell = ParagraphStyle("hc", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
data = [[Paragraph(str(c).replace("\n", "<br/>"), hcell if i == 0 else cell) if not isinstance(c, Paragraph) else c
         for c in row] for i, row in enumerate(data)]
tbl = Table(data, colWidths=[0.9*cm, 1.2*cm, 1.0*cm, 1.1*cm, 1.3*cm, 1.2*cm, 1.1*cm, 1.2*cm, 1.2*cm, 1.3*cm, 1.3*cm, 4.4*cm],
            repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
    ("BACKGROUND", (0, rows.index(best) + 1), (-1, rows.index(best) + 1), colors.HexColor("#fff3c4")),
]))
s.append(tbl)
s.append(Spacer(1, 4))
P("eta = I<sub>focus</sub> / mean background-sector intensity (the quantity the search maximises). "
  "eta vs random = optimised focus intensity / focus intensity averaged over random patterns. "
  f"The highlighted row is the kept result: run {best['run_id']}, eta = {best['enhancement']}.", cap)

P("5. Camera images", h2)
W = 4.2 * cm
grid = [[Paragraph("<b>Run</b>", cell), Paragraph("<b>Before (random pattern)</b>", cell),
         Paragraph("<b>After optimisation</b>", cell), Paragraph("<b>DMD pattern</b>", cell)]]
for r in rows:
    im = r["images"]
    grid.append([Paragraph(f'{r["run_id"]}<br/>m={r["settings"]["macropixel_size"]}<br/>eta={r["enhancement"]}', cell),
                 Image(im["initial_image"], W, W), Image(im["final_image"], W, W), Image(im["dmd_pattern"], W, W)])
g = Table(grid, colWidths=[1.8*cm, W + 0.3*cm, W + 0.3*cm, W + 0.3*cm], repeatRows=1)
g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
s.append(g)
P("Before and after images of each run share one colour scale (the after image's maximum), so the speckle before "
  "optimisation looks dim next to the focus. Cyan: focus region. White dashed: background annulus.", cap)

s.append(KeepTogether([Paragraph(f"Enhancement during the best run (run {best['run_id']})", h2),
                       Image(best["images"]["history"], 11*cm, 7.15*cm),
                       Paragraph("Kept enhancement against iteration. Steps are accepted flips. Small drops are re-measurements "
                                 "of the kept pattern that correct a noisy high shot.", cap)]))

P("6. Observations", h2)
by_m = {}
for r in rows:
    if r["settings"]["focus_size"] == 2 and r["settings"]["bg_inner"] == 4:
        by_m[r["settings"]["macropixel_size"]] = r
trend = ", ".join(f'm={m}: {by_m[m]["enhancement"]}' for m in sorted(by_m, reverse=True))
P(f"With the default regions the enhancement grows with the number of controlled macropixels ({trend}), as expected: "
  "every independent degree of freedom adds a field contribution that can be switched on when it interferes "
  "constructively at the focus. A 4 x 4 px focus region covers about four independent speckle grains. A binary pattern "
  "cannot bring all of them in phase at once, so the enhancement falls sharply. The tighter 3-8 px ring with 4 sectors "
  "gives a lower value than the 4-12 px ring, because the sidelobes of the focus spill into a ring that close.")
P("The maximised ratio (eta) can exceed N/(2 pi), while eta vs random stays below it. The ratio rewards both "
  "a brighter focus and a darker proximal background, and the search also dims the ring around the focus. "
  "N/(2 pi) is the gain in focus intensity alone over the random-pattern average, which is what eta vs random measures. "
  "The gap to theory in that column comes from three sources: shot noise in the comparisons, the finite number of "
  "iterations (the m = 1 run with 1024 macropixels was still climbing at the end), and greedy flips that settle in "
  "local optima.")

P("7. Comparison with the lecture", h2)
P("<i>[Insert the lecture's focusing result here: its enhancement value, number of segments, modulation type "
  "(phase or binary amplitude) and its camera image next to the image of run "
  f"{best['run_id']} above.]</i>")
P("Reference points from the literature for that comparison: Vellekoop and Mosk (2007) predict eta = (pi/4)(N-1)+1 for "
  "phase-only modulation and measured about 1000 with 3228 segments. For binary amplitude modulation with a DMD "
  "(Akbulut et al. 2011) the prediction is eta = N/(2 pi) + 1/2. A binary amplitude DMD therefore reaches about 2/pi^2, "
  f"roughly 20%, of the phase-only enhancement for the same N. Here the best run used N = {best['settings']['n_macropixels']} "
  f"macropixels and reached eta = {best['enhancement']} against the proximal background "
  f"(theory N/(2 pi) = {best['theory_binary_amplitude']}).")

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                        topMargin=1.6*cm, bottomMargin=1.6*cm, title="HW2 Task 2 report")
doc.build(s)
print("wrote", OUT)
