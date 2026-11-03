#!/usr/bin/env python3
"""Generate the README diagrams (light + dark SVG) into docs/img/.

    python3 scripts/readme-diagrams.py

Edit the layout here, not the SVG files. README.md embeds them with
<picture> so GitHub picks the variant for the viewer's color scheme.
"""
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

PALETTES = {
    "light": dict(paper="#f5f5f5", ink="#2d3142", muted="#4f5d75", soft="#7a8399",
                  rule="rgba(45,49,66,0.12)", accent="#eb6c36", tint="rgba(235,108,54,0.08)",
                  step="#ffffff", store="rgba(45,49,66,0.05)", input="rgba(79,93,117,0.10)",
                  zone="rgba(45,49,66,0.03)", zonestroke="rgba(45,49,66,0.25)"),
    "dark": dict(paper="#2d3142", ink="#f5f5f5", muted="#bfc0c0", soft="#8e98ac",
                 rule="rgba(245,245,245,0.12)", accent="#f08a59", tint="rgba(240,138,89,0.10)",
                 step="#393e53", store="rgba(245,245,245,0.05)", input="rgba(191,192,192,0.10)",
                 zone="rgba(245,245,245,0.03)", zonestroke="rgba(245,245,245,0.25)"),
}
SANS = "Geist, ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


class Svg:
    def __init__(self, slug, w, h, title, desc, c):
        self.c, self.parts = c, []
        self.head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="{slug}-title {slug}-desc">'
            f'<title id="{slug}-title">{escape(title)}</title><desc id="{slug}-desc">{escape(desc)}</desc>'
            '<defs>'
            f'<marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">'
            f'<polygon points="0 0, 8 3, 0 6" fill="{c["muted"]}"/></marker>'
            '</defs>'
            f'<rect width="{w}" height="{h}" rx="8" fill="{c["paper"]}"/>'
        )

    def text(self, x, y, s, size=12, weight=400, fill=None, mono=False, anchor="middle", spacing=None):
        sp = f' letter-spacing="{spacing}"' if spacing else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{fill or self.c["ink"]}" font-size="{size}" font-weight="{weight}" '
            f'font-family="{MONO if mono else SANS}" text-anchor="{anchor}"{sp}>{escape(s)}</text>')

    def path(self, d, dashed=False, head=True):
        dash = ' stroke-dasharray="5,4"' if dashed else ""
        end = ' marker-end="url(#arrow)"' if head else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{self.c["muted"]}" stroke-width="1.2"{dash}{end}/>')

    def label(self, cx, y_line, s):
        # ^ mask sits 8px above the connector so the line stays visible
        w = len(s) * 6 + 12
        self.parts.append(f'<rect x="{cx - w / 2}" y="{y_line - 20}" width="{w}" height="12" rx="2" fill="{self.c["paper"]}"/>')
        self.text(cx, y_line - 11, s, size=8, fill=self.c["soft"], mono=True, spacing="0.06em")

    def zone(self, x, y, w, h, name):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{self.c["zone"]}" '
                          f'stroke="{self.c["zonestroke"]}" stroke-width="1" stroke-dasharray="4,4"/>')
        self.text(x + 12, y + 16, name, size=9, fill=self.c["soft"], mono=True, anchor="start", spacing="0.06em")

    def box(self, x, y, w, h, kind, name, subs=(), tag=None):
        c = self.c
        fill, stroke, dash, rx = {
            "step": (c["step"], c["ink"], "", 6),
            "focal": (c["tint"], c["accent"], "", 6),
            "focal-opt": (c["tint"], c["accent"], ' stroke-dasharray="4,3"', 6),
            "store": (c["store"], c["muted"], "", 6),
            "input": (c["input"], c["soft"], "", 20),
        }[kind]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c["paper"]}"/>')
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1"{dash}/>')
        cx = x + w / 2
        if tag:
            tw = len(tag) * 5 + 12
            self.parts.append(f'<rect x="{x + 8}" y="{y + 8}" width="{tw}" height="12" rx="2" fill="none" '
                              f'stroke="{stroke}" stroke-opacity="0.4" stroke-width="0.8"/>')
            self.text(x + 8 + tw / 2, y + 17, tag, size=7, fill=stroke, mono=True, spacing="0.08em")
        lines = len(subs)
        name_y = y + h / 2 + 4 - lines * 7 + (6 if tag else 0)
        self.text(cx, name_y, name, size=12, weight=600)
        for i, s in enumerate(subs):
            self.text(cx, name_y + 16 + i * 13, s, size=9, fill=c["muted"], mono=True)

    def legend(self, y, w, items):
        c = self.c
        self.parts.append(f'<line x1="20" y1="{y}" x2="{w - 20}" y2="{y}" stroke="{c["rule"]}" stroke-width="0.8"/>')
        self.text(20, y + 24, "LEGEND", size=8, fill=c["muted"], mono=True, anchor="start", spacing="0.14em")
        x = 100
        for kind, s in items:
            if kind in ("arrow", "dashed"):
                self.path(f"M{x},{y + 21} H{x + 28}", dashed=kind == "dashed")
            else:
                stroke = c["accent"] if kind == "focal" else c["muted"]
                fill = c["tint"] if kind == "focal" else c["store"]
                self.parts.append(f'<rect x="{x}" y="{y + 15}" width="28" height="12" rx="2" fill="{fill}" stroke="{stroke}"/>')
            self.text(x + 36, y + 24, s, size=9, fill=c["muted"], mono=True, anchor="start")
            x += 52 + len(s) * 6

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text(self.head + "".join(self.parts) + "</svg>\n")


def search_pipeline(variant):
    s = Svg(f"search-pipeline-{variant}", 960, 352, "ir search pipeline",
            "A query goes through BM25 at tier 0, then BM25 and vector search in parallel with a "
            "per-query fusion at tier 1, then a reranker at tier 2; tiers 0 and 1 return early when confident.",
            PALETTES[variant])
    # arrows first, boxes paint over their ends
    for x1, x2 in ((116, 140), (300, 324), (484, 508), (668, 692), (820, 844)):
        s.path(f"M{x1},136 H{x2}")
    s.path("M220,96 V48 Q220,40 228,40 H908 Q916,40 916,48 V96", dashed=True)
    s.label(420, 40, "STRONG MATCH")
    s.path("M588,96 V80 Q588,72 596,72 H860 Q868,72 868,80 V96", dashed=True)
    s.label(756, 72, "CONFIDENT")
    s.path("M548,176 V188 Q548,196 540,196 H420 Q412,196 412,204 V216")
    s.path("M588,176 V216")
    s.path("M628,176 V188 Q628,196 636,196 H756 Q764,196 764,204 V216")

    s.box(20, 116, 96, 40, "input", "Query")
    s.box(140, 96, 160, 80, "step", "BM25", ("FTS5 + doc graph", "no model · ~1 ms"), tag="TIER 0")
    s.box(324, 96, 160, 80, "step", "BM25 ∥ Vector", ("run in parallel", "HNSW · ~50 ms"), tag="TIER 1")
    s.box(508, 96, 160, 80, "focal", "Per-query fusion", ("picks a weighting", "new in 0.19"), tag="TIER 1")
    s.box(692, 96, 128, 80, "step", "Rerank", ("reranker 0.6B", "top 100 · ~2–5 s"), tag="TIER 2")
    s.box(844, 116, 96, 40, "input", "Results")
    s.box(332, 216, 160, 64, "store", "Keyword", ("vector + BM25 on exact hits",), tag="ALL WORDS IN TOP-3 BM25")
    s.box(508, 216, 160, 64, "store", "Cross-language", ("vector only",), tag="HITS IN ANOTHER SCRIPT")
    s.box(684, 216, 160, 64, "store", "Default mix", ("0.8 vector + 0.2 BM25",), tag="OTHERWISE")
    s.legend(304, 960, [("arrow", "escalate"), ("dashed", "return early"),
                        ("focal", "new in 0.19"), ("store", "fusion choice")])
    s.save(f"search-pipeline-{variant}.svg")


def index_lifecycle(variant):
    s = Svg(f"index-lifecycle-{variant}", 960, 424, "What ir sync builds",
            "ir update reads markdown files into a BM25 index and, with --code-glob, code symbols; "
            "ir embed turns the indexed text into vectors, a doc graph and an HNSW ANN index.",
            PALETTES[variant])
    s.zone(356, 20, 368, 336, "notes.sqlite")
    s.text(164, 36, "IR SYNC = UPDATE + EMBED", size=8, fill=s.c["soft"], mono=True, anchor="start", spacing="0.1em")
    s.path("M140,80 H164")
    s.path("M324,80 H380")
    s.path("M324,168 H380")
    s.path("M324,272 H380")
    s.path("M524,272 H556")
    s.path("M452,304 V312 Q452,320 460,320 H852 Q860,320 860,312 V304")

    s.box(20, 48, 120, 64, "input", "Markdown files", ("~/notes/**/*.md",))
    s.box(164, 48, 160, 152, "step", "ir update", ("read · preprocess", "ko / ja / zh", "tree-sitter for code"), tag="NO MODEL")
    s.box(164, 236, 160, 72, "step", "ir embed", ("chunk · embed",), tag="EMBEDDING MODEL")
    s.box(380, 48, 320, 64, "store", "BM25 index", ("FTS5",), tag="STORE")
    s.box(380, 136, 320, 64, "focal-opt", "Code symbols", ("only with --code-glob",), tag="OPTIONAL")
    s.box(380, 240, 144, 64, "store", "Vectors", ("sqlite-vec",), tag="STORE")
    s.box(556, 240, 144, 64, "store", "Doc graph", ("kNN neighbours",), tag="STORE")
    s.box(780, 240, 160, 64, "store", "HNSW ANN index", ("notes.usearch",), tag="FILE")
    s.text(244, 332, "reuses cached vectors", size=9, fill=s.c["soft"], mono=True)
    s.legend(376, 960, [("arrow", "writes"), ("focal", "new in 0.19 (optional)"), ("store", "stored index")])
    s.save(f"index-lifecycle-{variant}.svg")


if __name__ == "__main__":
    for v in PALETTES:
        search_pipeline(v)
        index_lifecycle(v)
    print("wrote", *sorted(p.name for p in OUT.glob("*.svg")))
