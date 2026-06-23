"""
scripts/eval_queries.py — run test queries and generate an HTML quality report.

Usage:
    python -m scripts.eval_queries
    python -m scripts.eval_queries --top 5
    python -m scripts.eval_queries --out /tmp/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import sys
from datetime import datetime
from pathlib import Path

from app import config
from app.search import SearchResult, search

config.ensure_dirs()

QUERIES = [
    "кот орёт",
    "я устал",
    "ну пиздец",
    "сарказм",
    "шок",
    "обнимаю",
    "злой",
    "смешно но больно",
    "кринж",
    "плач",
    "паника",
    "радость",
    "победа",
    "страшно",
    "спасибо",
    "иди нахуй",
    "сонный",
    "уверенность",
]


def _thumb_src(result: SearchResult) -> str:
    """Return an inline base64 data-URI for the best preview frame, or ''."""
    conn = __import__("app.db", fromlist=["get_conn"]).get_conn()
    row = conn.execute(
        """
        SELECT mf.preview_path
        FROM media_frames mf
        JOIN frame_embeddings fe ON fe.frame_id = mf.id
        JOIN media_items mi ON mi.id = mf.media_id
        WHERE mi.id = ?
        ORDER BY mf.frame_index
        LIMIT 1
        """,
        (result.media_id,),
    ).fetchone()
    if not row:
        return ""
    path = Path(row["preview_path"])
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def _card(result: SearchResult) -> str:
    thumb = _thumb_src(result)
    img_tag = (
        f'<img src="{thumb}" alt="preview">'
        if thumb
        else '<div class="no-thumb">no preview</div>'
    )
    badges = []
    if result.is_favorite:
        badges.append('<span class="badge fav">fav</span>')
    if result.is_recent:
        badges.append('<span class="badge rec">recent</span>')
    set_name = html.escape(result.set_title or result.set_short_name or "—")
    emoji = html.escape(result.emoji or "")
    kind = html.escape(result.sticker_format or result.media_kind)
    return f"""<div class="card">
  {img_tag}
  <div class="meta">
    <div class="score">{result.score:.4f}</div>
    <div class="kind">{kind} {emoji}</div>
    <div class="set" title="{html.escape(result.set_short_name or '')}">{set_name}</div>
    {"".join(badges)}
  </div>
</div>"""


def _section(query: str, results: list[SearchResult]) -> str:
    cards = "".join(_card(r) for r in results) if results else "<p>no results</p>"
    return f"""<section>
  <h2>{html.escape(query)}</h2>
  <div class="grid">{cards}</div>
</section>"""


_CSS = """
body { font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 1rem; }
h1   { font-size: 1.4rem; margin-bottom: 1rem; }
h2   { font-size: 1rem; color: #aaa; border-bottom: 1px solid #333; padding-bottom: .3rem; margin: 1.5rem 0 .6rem; }
.grid { display: flex; flex-wrap: wrap; gap: .5rem; }
.card { background: #1e1e1e; border-radius: 6px; padding: .4rem; width: 120px; text-align: center; }
.card img { width: 100px; height: 100px; object-fit: contain; background: #fff; border-radius: 4px; }
.no-thumb { width: 100px; height: 100px; background: #333; display: flex; align-items: center; justify-content: center; font-size: .65rem; color: #777; margin: 0 auto; border-radius: 4px; }
.meta { font-size: .65rem; margin-top: .3rem; }
.score { font-weight: bold; color: #7ec8e3; }
.kind  { color: #aaa; }
.set   { color: #ccc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.badge { display: inline-block; font-size: .55rem; border-radius: 3px; padding: 0 3px; margin: 1px; }
.fav   { background: #b8860b; }
.rec   { background: #2e6b2e; }
"""


def generate(top_k: int) -> str:
    sections = []
    for q in QUERIES:
        print(f"  {q!r} ...", end=" ", flush=True)
        results = search(q, top_k=top_k)
        print(f"{len(results)} results")
        sections.append(_section(q, results))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>StickerRadar eval {ts}</title>
<style>{_CSS}</style></head>
<body>
<h1>StickerRadar — eval report &nbsp;<small style="color:#666">{ts} · top {top_k}</small></h1>
{body}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    out_path = (
        Path(args.out)
        if args.out
        else config.EVAL_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )

    print(f"Running {len(QUERIES)} queries (top {args.top}) ...")
    report = generate(top_k=args.top)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
