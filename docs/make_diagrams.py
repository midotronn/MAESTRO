#!/usr/bin/env python3
"""Generate MAESTRO's method diagrams as clean, on-brand SVGs (light theme, purple palette).

Two focused figures, emphasising the AGENTIC framework and de-emphasising rendering:
  framework.svg     - the composition pipeline: audio + structure -> Storyboard Agent (LLM)
                      -> assemble LODGE/EDGE + motifs -> a structured dance.
  editing_loop.svg  - the natural-language editing agent: the Planner -> Executor -> Verify loop,
                      with the refine cycle, the motion toolset, and the no-regression guardrail.

Run:  python docs/make_diagrams.py   (writes docs/static/images/*.svg)
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "static" / "images"

# ---- MAESTRO palette (matches static/css/style.css) ----
INK = "#1a2030"; MUTED = "#5a6272"; FAINT = "#8b93a7"
LINE = "#e4e7f0"; CARD = "#ffffff"; SOFT = "#f5f6fb"
BRAND = "#6c4ce0"; BRAND2 = "#9b3fd4"; ACCENT = "#1aa79a"
LODGE = "#3b82f6"; EDGE = "#9333ea"; REUSE = "#e0870a"
AGENT = "#f3effe"           # light-purple agent fill
FONT = "system-ui,'Segoe UI',Roboto,'Noto Sans',sans-serif"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=14, fill=INK, weight=600, anchor="middle", italic=False):
    it = ' font-style="italic"' if italic else ''
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{it}>{esc(s)}</text>')


def rrect(x, y, w, h, fill=CARD, stroke=LINE, sw=1.5, rx=14):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')


def node(x, y, w, h, title, subs=None, fill=CARD, stroke=LINE, accent=None,
         title_size=14.5, title_fill=INK, badge=None, sub_fill=MUTED):
    p = [rrect(x, y, w, h, fill, stroke)]
    if accent:
        p.append(f'<rect x="{x}" y="{y}" width="6" height="{h}" rx="3" fill="{accent}"/>')
        p.append(f'<rect x="{x + 3}" y="{y}" width="3" height="{h}" fill="{fill}"/>')
    cx = x + w / 2
    ty = y + (h / 2 - 8 * (len(subs) if subs else 0) + 5) if subs else y + h / 2 + 5
    if badge:
        p.append(txt(x + w / 2, y + 20, badge, 9.5, BRAND, 700))
        ty = y + 20
    if subs:
        p.append(txt(cx, ty + (14 if badge else 0), title, title_size, title_fill, 700))
        for i, s in enumerate(subs):
            p.append(txt(cx, ty + (14 if badge else 0) + 18 + i * 15, s, 11, sub_fill, 500))
    else:
        p.append(txt(cx, y + h / 2 + 5, title, title_size, title_fill, 700))
    return "".join(p)


def arrow(x1, y1, x2, y2, color=MUTED, label=None, dashed=False, cy=None):
    dash = ' stroke-dasharray="5 4"' if dashed else ''
    if cy is None:
        d = f'M {x1} {y1} L {x2} {y2}'
        lx, ly = (x1 + x2) / 2, min(y1, y2) - 8
    else:  # smooth curve bowing to y=cy at the midpoint
        mx = (x1 + x2) / 2
        d = f'M {x1} {y1} C {mx} {cy} {mx} {cy} {x2} {y2}'
        lx, ly = mx, cy + (14 if cy > y1 else -6)
    out = (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} '
           f'marker-end="url(#ah)"/>')
    if label:
        out += txt(lx, ly, label, 10.5, MUTED, 600)
    return out


DEFS = (
    '<defs>'
    '<marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
    f'orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 z" fill="{MUTED}"/></marker>'
    f'<linearGradient id="brandg" x1="0" y1="0" x2="1" y2="1">'
    f'<stop offset="0" stop-color="{BRAND}"/><stop offset="1" stop-color="{BRAND2}"/></linearGradient>'
    '</defs>'
)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="100%" role="img">{DEFS}<rect width="{w}" height="{h}" fill="{CARD}"/>'
            f'{body}</svg>\n')


# ============================================================ framework.svg
def framework():
    W, H = 1160, 300
    b = []
    b.append(txt(W / 2, 34, "The MAESTRO composition pipeline", 17, INK, 800))
    b.append(txt(W / 2, 55, "an LLM storyboard agent turns musical structure into a composed, motif-aware dance",
                 12.5, MUTED, 500))
    yr = 92; h = 84
    # nodes
    b.append(node(28, yr + 12, 96, 60, "Song", ["audio"], fill=SOFT, title_size=14))
    b.append(node(150, yr, 168, h, "Audio & structure",
                  ["features · sections", "energy arc · downbeats"]))
    b.append(node(350, yr, 216, h, "Storyboard Agent", ["a per-section plan: role,", "intensity, vocabulary, motif"],
                  fill=AGENT, stroke=BRAND, accent=BRAND, title_fill=BRAND, badge="LLM AGENT"))
    b.append(node(598, yr, 214, h, "Structure-aware assembly",
                  ["arrange LODGE + EDGE +", "motif reuse · seamless joins"]))
    b.append(node(844, yr + 8, 150, 68, "Structured dance", None, fill="url(#brandg)",
                  stroke=BRAND, title_fill="#ffffff", title_size=14))
    # generator chips under assembly
    for i, (lbl, col) in enumerate([("LODGE", LODGE), ("EDGE", EDGE), ("motif ↺", REUSE)]):
        cx = 606 + i * 68
        b.append(f'<rect x="{cx}" y="{yr + h + 12}" width="62" height="22" rx="11" fill="{CARD}" stroke="{col}"/>')
        b.append(txt(cx + 31, yr + h + 27, lbl, 10.5, col, 700))
    cy = yr + h / 2
    b.append(arrow(124, cy, 150, cy))
    b.append(arrow(318, cy, 350, cy))
    b.append(arrow(566, cy, 598, cy))
    b.append(arrow(812, cy, 844, cy))
    return svg(W, H, "".join(b))


# ============================================================ editing_loop.svg
def editing_loop():
    W, H = 1160, 340
    b = []
    b.append(txt(W / 2, 34, "The natural-language editing agent", 17, INK, 800))
    b.append(txt(W / 2, 55, "a planner-executor loop refines any window until it meets the goals you asked for",
                 12.5, MUTED, 500))
    yr = 108; h = 92
    b.append(node(28, yr, 196, h, "You", ["pick a time window +", "describe the change"],
                  fill=SOFT))
    b.append(txt(126, yr + h + 20, "\u201ccalmer, but keep it", 11, FAINT, 500, italic=True))
    b.append(txt(126, yr + h + 34, "tight to the beat\u201d", 11, FAINT, 500, italic=True))
    b.append(node(252, yr, 210, h, "Planner", ["declare the goals,", "plan the tool steps"],
                  fill=AGENT, stroke=BRAND, accent=BRAND, title_fill=BRAND, badge="LLM"))
    b.append(node(490, yr, 210, h, "Executor", ["apply the motion tools,", "reject any regression"],
                  fill=CARD, stroke=LINE, accent=ACCENT, title_fill=INK))
    b.append(node(728, yr, 174, h, "Verify", ["measure every", "requested goal"]))
    b.append(node(930, yr + 10, 156, 72, "Refined window", ["+ checkpoint"], fill="url(#brandg)",
                  stroke=BRAND, title_fill="#ffffff", title_size=13.5, sub_fill="#ede9fb"))
    cy = yr + h / 2
    b.append(arrow(224, cy, 252, cy))
    b.append(arrow(462, cy, 490, cy))
    b.append(arrow(700, cy, 728, cy))
    b.append(arrow(902, cy, 930, cy + 5, color=ACCENT, label="goals met"))
    # refine loop: verify top -> planner top (bows upward); label sits in the gap below the arc
    b.append(arrow(815, yr, 357, yr, color=BRAND, dashed=True, cy=yr - 44))
    b.append(txt(586, yr - 12, "short of goal \u2192 refine (with feedback)", 10.5, BRAND, 600))
    # toolset chip row under executor
    tools = ["energy", "beat-align", "smooth", "sharpen", "mirror", "retrograde", "regenerate"]
    tx = 300; ty = yr + h + 40
    b.append(txt(tx - 6, ty + 15, "tools", 10.5, FAINT, 700, anchor="end"))
    cx = tx + 12
    for t in tools:
        wdt = 12 + len(t) * 6.4
        b.append(f'<rect x="{cx}" y="{ty + 3}" width="{wdt:.0f}" height="22" rx="11" fill="{SOFT}" stroke="{LINE}"/>')
        b.append(txt(cx + wdt / 2, ty + 18, t, 10.5, MUTED, 600))
        cx += wdt + 8
    b.append(txt(tx + 12, ty + 44, "guardrail: no step may worsen the goal it targets; on failure the plan is refined, not shipped",
                 10.5, FAINT, 500, anchor="start"))
    return svg(W, H, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "framework.svg").write_text(framework(), encoding="utf-8")
    (OUT / "editing_loop.svg").write_text(editing_loop(), encoding="utf-8")
    print("wrote", OUT / "framework.svg", "and", OUT / "editing_loop.svg")


if __name__ == "__main__":
    main()
