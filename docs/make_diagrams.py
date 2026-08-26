#!/usr/bin/env python3
"""Generate MAESTRO's method diagrams as clean, on-brand SVGs (light theme, purple palette).

Two focused figures, emphasizing the agentic framework and de-emphasizing rendering:
  framework.svg     - deterministic music analysis -> Storyboard Agent (LLM or fallback)
                      -> plan-aware realization from LODGE/EDGE/reuse plus named-motion cues
                      -> inertialized assembly.
  editing_loop.svg  - Planner -> routed tools -> final-window adaptation -> bounded verification,
                      refinement, and the best safe checkpointed output.

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
    W, H = 1160, 330
    b = []
    b.append(txt(W / 2, 34, "The MAESTRO composition pipeline", 17, INK, 800))
    b.append(txt(W / 2, 55, "musical form guides source selection, motif reuse, and planned named-motion cues",
                 12.5, MUTED, 500))
    yr = 92; h = 94
    b.append(node(20, yr + 17, 82, 60, "Song", ["audio"], fill=SOFT, title_size=14))
    b.append(node(126, yr, 186, h, "Music analysis",
                  ["chroma + MFCC + energy", "sections · repeats · downbeats"],
                  accent=ACCENT, title_size=14))
    b.append(node(338, yr, 204, h, "Storyboard Agent",
                  ["role · intensity · source bias", "motif reuse · named-motion cues"],
                  fill=AGENT, stroke=BRAND, accent=BRAND, title_fill=BRAND, badge="LLM AGENT"))
    b.append(node(568, yr, 244, h, "Plan-aware realization",
                  ["score LODGE · EDGE · reuse", "compose planned motion cues"],
                  accent=REUSE, title_size=14))
    b.append(node(838, yr, 168, h, "Inertialized assembly",
                  ["join chosen sections", "without hard source seams"], title_size=13.5))
    b.append(node(1032, yr + 11, 112, 72, "Structured", ["dance"], fill="url(#brandg)",
                  stroke=BRAND, title_fill="#ffffff", title_size=14, sub_fill="#ede9fb"))
    # Candidate sources plus the planned overlay applied by the realization stage.
    pills = [
        ("LODGE", LODGE, 54),
        ("EDGE", EDGE, 54),
        ("motif ↺", REUSE, 64),
        ("bank cue +", ACCENT, 72),
    ]
    cx = 559
    for lbl, col, width in pills:
        b.append(
            f'<rect x="{cx}" y="{yr + h + 13}" width="{width}" height="22" '
            f'rx="11" fill="{CARD}" stroke="{col}"/>'
        )
        b.append(txt(cx + width / 2, yr + h + 28, lbl, 10.5, col, 700))
        cx += width + 6
    b.append(txt(219, yr + h + 28, "deterministic, with a robust fallback", 10.5, FAINT, 500))
    b.append(txt(440, yr + h + 28, "strict section plan; rule fallback offline", 10.5, FAINT, 500))
    cy = yr + h / 2
    b.append(arrow(102, cy, 126, cy))
    b.append(arrow(312, cy, 338, cy))
    b.append(arrow(542, cy, 568, cy))
    b.append(arrow(812, cy, 838, cy))
    b.append(arrow(1006, cy, 1032, cy))
    return svg(W, H, "".join(b))


# ============================================================ editing_loop.svg
def editing_loop():
    W, H = 1160, 410
    b = []
    b.append(txt(W / 2, 34, "The natural-language editing agent", 17, INK, 800))
    b.append(txt(W / 2, 55, "monotone levers, audited named actions, and generation share a bounded verified loop",
                 12.5, MUTED, 500))
    yr = 112; h = 96
    b.append(node(18, yr, 172, h, "You", ["select a window +", "describe the change"],
                  fill=SOFT))
    b.append(node(216, yr, 190, h, "Planner", ["declare measurable goals", "+ choose a tool path"],
                  fill=AGENT, stroke=BRAND, accent=BRAND, title_fill=BRAND, badge="LLM"))
    b.append(node(432, yr, 228, h, "Routed executor",
                  ["metric intent: monotone levers", "named action: audited motion bank",
                   "new motion: LODGE / EDGE"],
                  fill=CARD, stroke=LINE, accent=ACCENT, title_fill=INK))
    b.append(node(686, yr, 202, h, "Final-window adaptation",
                  ["fit or crossfade into dance", "dial regenerated goals to target"],
                  accent=REUSE, title_size=13.5))
    b.append(node(914, yr, 152, h, "Verify",
                  ["goals + semantics", "quality guardrails"]))
    b.append(node(968, 302, 174, 68, "Checkpointed result", ["undo · compare · branch"],
                  fill="url(#brandg)", stroke=BRAND, title_fill="#ffffff",
                  title_size=13.5, sub_fill="#ede9fb"))
    cy = yr + h / 2
    b.append(arrow(190, cy, 216, cy))
    b.append(arrow(406, cy, 432, cy))
    b.append(arrow(660, cy, 686, cy))
    b.append(arrow(888, cy, 914, cy))
    b.append(arrow(990, yr + h, 1055, 302, color=ACCENT, label="best safe result", cy=270))
    # Refine loop feeds failed final-window checks back to the planner.
    b.append(arrow(990, yr, 311, yr, color=BRAND, dashed=True, cy=yr - 46))
    b.append(txt(650, yr - 13, "unmet goal or artifact risk \u2192 bounded refine with measured feedback",
                 10.5, BRAND, 600))
    # Toolset and execution contracts.
    tools = ["energy", "beat-align", "smooth", "sharpen", "mirror", "reverse", "regenerate"]
    tx = 270; ty = yr + h + 38
    b.append(txt(tx - 6, ty + 15, "tools", 10.5, FAINT, 700, anchor="end"))
    cx = tx + 12
    for t in tools:
        wdt = 12 + len(t) * 6.4
        b.append(f'<rect x="{cx}" y="{ty + 3}" width="{wdt:.0f}" height="22" rx="11" fill="{SOFT}" stroke="{LINE}"/>')
        b.append(txt(cx + wdt / 2, ty + 18, t, 10.5, MUTED, 600))
        cx += wdt + 8
    b.append(txt(tx + 12, ty + 47,
                 "verification measures the final spliced motion; recognized named actions receive semantic checks and failed steps stay visible",
                 10.5, FAINT, 500, anchor="start"))
    return svg(W, H, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "framework.svg").write_text(framework(), encoding="utf-8")
    (OUT / "editing_loop.svg").write_text(editing_loop(), encoding="utf-8")
    print("wrote", OUT / "framework.svg", "and", OUT / "editing_loop.svg")


if __name__ == "__main__":
    main()
