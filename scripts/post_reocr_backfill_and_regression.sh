#!/usr/bin/env bash
set -euo pipefail
PID="$1"
REPO="/home/jar/Documents/projects/StickerRadar"
cd "$REPO"
while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done

uv run python -m app sync --ocr-text-embed --yes

python - <<'PY'
from pathlib import Path
from datetime import datetime
from app.search import search
queries = [
    'я тупой',
    'я чувствую себя плохо',
    'мне плохо',
    'я идиот',
    'я в депрессии',
    'грустный кот',
    'человек орет',
    'совсем случайный бессмысленный запрос',
]
lines = ['# Live Search Regression Report', '', f'Generated: {datetime.now().isoformat(timespec="seconds")}', '']
for q in queries:
    lines.append(f'## Query: `{q}`')
    results = search(q, top_k=5)
    if not results:
        lines.append('- NO RESULTS')
        lines.append('')
        continue
    for i, r in enumerate(results[:5], 1):
        lines.append(f'- {i}. media_id={r.media_id} score={r.score:.4f} set={r.set_title or r.set_short_name or "-"} emoji={r.emoji or "-"} doc={r.tg_document_id}')
    lines.append('')
report = Path('docs/live-search-regression-report.md')
report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(report)
PY
