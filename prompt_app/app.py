#!/usr/bin/env python3
"""Small CSV-backed web app for authoring independent free-form prompts.

The app intentionally has no third-party dependencies.  It edits
free_form_pku.csv and free_form_cgl.csv directly so the existing paper scripts
can validate and consume the finished prompt files without conversion.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlencode, urlparse


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = APP_DIR / "config.json"
CSV_LOCK = threading.Lock()


DATASET_CLASS_GUIDANCE = {
    # Must match code/cgbdm/text_spatial.py and code/generate_prompts.py.
    # Class 0 is padding/no-element and is intentionally not shown to authors.
    "pku": [
        ("Text", ("text", "text box")),
        ("Logo", ("logo", "icon", "brand mark")),
        ("Underlay", ("underlay", "panel", "background panel")),
    ],
    "cgl": [
        ("Text", ("text", "text box")),
        ("Logo", ("logo", "icon", "brand mark")),
        ("Underlay", ("underlay", "panel", "background panel")),
        ("Embellishment", ("embellishment", "decoration", "graphic element")),
    ],
}


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    label: str
    csv_path: Path
    image_dir: Path


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    backup_on_save: bool
    datasets: dict[str, DatasetConfig]


def resolve_path(value: str, base: Path = APP_DIR) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_config(path: Path) -> AppConfig:
    raw = json.loads(path.read_text())
    datasets: dict[str, DatasetConfig] = {}
    for key, item in raw.get("datasets", {}).items():
        datasets[key] = DatasetConfig(
            key=key,
            label=item.get("label", key.upper()),
            csv_path=resolve_path(item["csv"], path.parent),
            image_dir=resolve_path(item["image_dir"], path.parent),
        )
    if not datasets:
        raise ValueError("config.json must define at least one dataset")
    return AppConfig(
        host=raw.get("host", "0.0.0.0"),
        port=int(raw.get("port", 7860)),
        backup_on_save=bool(raw.get("backup_on_save", True)),
        datasets=datasets,
    )


def read_rows(dataset: DatasetConfig) -> tuple[list[str], list[dict[str, str]], str]:
    if not dataset.csv_path.exists():
        raise FileNotFoundError(dataset.csv_path)
    with dataset.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [{key: value or "" for key, value in row.items()} for row in reader]
    prompt_column = "text_prompt" if "text_prompt" in fieldnames else "prompt"
    required = {"poster_path", prompt_column}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{dataset.csv_path} missing required columns: {sorted(missing)}")
    for optional in ("author_id", "independent_of_ground_truth"):
        if optional not in fieldnames:
            fieldnames.append(optional)
            for row in rows:
                row[optional] = "yes" if optional == "independent_of_ground_truth" else ""
    return fieldnames, rows, prompt_column


def atomic_write_csv(dataset: DatasetConfig, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    dataset.csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dataset.csv_path.name}.", suffix=".tmp", dir=str(dataset.csv_path.parent)
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(tmp_name, dataset.csv_path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def make_backup(dataset: DatasetConfig) -> None:
    if not dataset.csv_path.exists():
        return
    backup_dir = APP_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"{dataset.csv_path.stem}_{stamp}.csv"
    # Avoid overwriting if two saves happen within one second.
    counter = 1
    while backup.exists():
        backup = backup_dir / f"{dataset.csv_path.stem}_{stamp}_{counter}.csv"
        counter += 1
    shutil.copy2(dataset.csv_path, backup)


def row_is_blank(row: dict[str, str], prompt_column: str) -> bool:
    return not row.get(prompt_column, "").strip()


def row_matches(row: dict[str, str], prompt_column: str, query: str) -> bool:
    if not query:
        return True
    blob = " ".join(
        [
            row.get("poster_path", ""),
            row.get(prompt_column, ""),
            row.get("author_id", ""),
            row.get("independent_of_ground_truth", ""),
        ]
    ).lower()
    return query.lower() in blob


def filtered_indices(
    rows: list[dict[str, str]], prompt_column: str, filter_name: str, query: str
) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(rows):
        if filter_name == "blank" and not row_is_blank(row, prompt_column):
            continue
        if filter_name == "done" and row_is_blank(row, prompt_column):
            continue
        if not row_matches(row, prompt_column, query):
            continue
        indices.append(index)
    return indices


def dataset_class_guidance_html(dataset: DatasetConfig) -> str:
    classes = DATASET_CLASS_GUIDANCE.get(dataset.key.lower(), DATASET_CLASS_GUIDANCE["cgl"])
    lines = []
    for canonical, aliases in classes:
        alias_html = ", ".join(f"<code>{html.escape(alias)}</code>" for alias in aliases)
        lines.append(f"<li><b>{html.escape(canonical)}</b>: {alias_html}</li>")
    return "\n".join(lines)


def dataset_extra_guidance_html(dataset: DatasetConfig) -> str:
    lines = [
        "<li>Keep prompts natural, but keep class words and spatial keywords clear enough "
        "for the model to read.</li>",
        "<li>If describing titles or captions, include the class word too, e.g. "
        "<code>one text box for the title</code> or <code>caption text at bottom-center</code>.</li>",
        "<li>Mild natural variation is fine; avoid spelling mistakes in key words such as "
        "<code>text</code>, <code>logo</code>, <code>underlay</code>, and "
        "<code>top-center</code>.</li>",
    ]
    if dataset.key.lower() == "pku":
        lines.append(
            "<li>Avoid <code>decoration</code>, <code>embellishment</code>, and "
            "<code>graphic element</code>; that class exists only in CGL.</li>"
        )
    elif dataset.key.lower() == "cgl":
        lines.append(
            "<li>CGL may use <code>embellishment</code>, <code>decoration</code>, or "
            "<code>graphic element</code> for decorative visual objects.</li>"
        )
    return "\n".join(lines)


def safe_image_path(dataset: DatasetConfig, poster_path: str) -> Path | None:
    # Allow nested poster paths if a dataset ever uses them, but prevent traversal.
    relative = Path(unquote(poster_path))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (dataset.image_dir / relative).resolve()
    try:
        candidate.relative_to(dataset.image_dir)
    except ValueError:
        return None
    return candidate


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def redirect(handler: BaseHTTPRequestHandler, location: str) -> None:
    handler.send_response(HTTPStatus.SEE_OTHER)
    handler.send_header("Location", location)
    handler.end_headers()


def render_page(
    cfg: AppConfig,
    dataset: DatasetConfig,
    selected_index: int | None,
    filter_name: str,
    query: str,
) -> str:
    fieldnames, rows, prompt_column = read_rows(dataset)
    total = len(rows)
    filled = sum(not row_is_blank(row, prompt_column) for row in rows)
    blank = total - filled
    indices = filtered_indices(rows, prompt_column, filter_name, query)
    if selected_index is None or selected_index not in indices:
        selected_index = indices[0] if indices else None
    selected_position = indices.index(selected_index) if selected_index in indices else -1
    row = rows[selected_index] if selected_index is not None else {}
    poster_path = row.get("poster_path", "")
    prompt = row.get(prompt_column, "")
    author_id = row.get("author_id", "")
    independent = row.get("independent_of_ground_truth", "yes") or "yes"
    class_guidance = dataset_class_guidance_html(dataset)
    extra_guidance = dataset_extra_guidance_html(dataset)
    image_url = (
        f"/image/{dataset.key}/{poster_path}"
        if poster_path
        else ""
    )

    nav_links = []
    for key, item in cfg.datasets.items():
        active = "active" if key == dataset.key else ""
        nav_links.append(
            f'<a class="dataset {active}" href="/dataset/{html.escape(key)}?filter=blank">'
            f"{html.escape(item.label)}</a>"
        )

    def filter_link(name: str, label: str) -> str:
        params = urlencode({"filter": name, "q": query})
        active = "active" if name == filter_name else ""
        return f'<a class="chip {active}" href="/dataset/{dataset.key}?{params}">{label}</a>'

    previous_link = "#"
    next_link = "#"
    if selected_position > 0:
        prev_index = indices[selected_position - 1]
        previous_link = (
            f"/dataset/{dataset.key}?{urlencode({'index': prev_index, 'filter': filter_name, 'q': query})}"
        )
    if 0 <= selected_position < len(indices) - 1:
        next_index = indices[selected_position + 1]
        next_link = (
            f"/dataset/{dataset.key}?{urlencode({'index': next_index, 'filter': filter_name, 'q': query})}"
        )

    row_links = []
    for pos, idx in enumerate(indices[:500]):
        item = rows[idx]
        item_prompt = item.get(prompt_column, "")
        status = "done" if item_prompt.strip() else "blank"
        active = "active" if idx == selected_index else ""
        label = html.escape(item.get("poster_path", f"row {idx + 1}"))
        row_links.append(
            f'<a class="row-link {active} {status}" '
            f'href="/dataset/{dataset.key}?{urlencode({"index": idx, "filter": filter_name, "q": query})}">'
            f'<span>{idx + 1:03d}</span><b>{label}</b></a>'
        )
    if len(indices) > 500:
        row_links.append('<div class="muted small">Showing first 500 matching rows.</div>')

    escaped_prompt = html.escape(prompt)
    escaped_author = html.escape(author_id)
    escaped_independent = html.escape(independent)
    escaped_poster = html.escape(poster_path)
    escaped_query = html.escape(query)
    progress = round((filled / total) * 100, 1) if total else 0.0

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt Authoring — {html.escape(dataset.label)}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111c33;
      --panel2: #17233d;
      --text: #eef4ff;
      --muted: #9fb0ce;
      --accent: #60a5fa;
      --good: #34d399;
      --warn: #fbbf24;
      --danger: #fb7185;
      --border: #263855;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(135deg, #0f172a, #111827 55%, #0b1120);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: inherit; text-decoration: none; }}
    .app {{ min-height: 100vh; display: grid; grid-template-columns: 320px 1fr; }}
    aside {{
      border-right: 1px solid var(--border);
      background: rgba(17, 28, 51, 0.94);
      padding: 18px;
      overflow-y: auto;
      max-height: 100vh;
      position: sticky;
      top: 0;
    }}
    main {{ padding: 22px; }}
    .brand {{ font-size: 20px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 14px; }}
    .datasets {{ display: flex; gap: 8px; margin-bottom: 14px; }}
    .dataset, .chip, button, .navbtn {{
      border: 1px solid var(--border);
      background: var(--panel2);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
      cursor: pointer;
    }}
    .dataset.active, .chip.active {{ border-color: var(--accent); background: #1d4ed8; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin: 12px 0;
    }}
    .stat {{ background: var(--panel2); border: 1px solid var(--border); border-radius: 12px; padding: 10px; }}
    .stat b {{ display: block; font-size: 20px; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 12px; }}
    .progress {{ height: 8px; background: #0b1220; border-radius: 999px; overflow: hidden; margin: 10px 0 16px; }}
    .bar {{ height: 100%; width: {progress}%; background: linear-gradient(90deg, var(--accent), var(--good)); }}
    .filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .search {{ display: flex; gap: 8px; margin: 12px 0; }}
    input, textarea, select {{
      width: 100%;
      background: #081121;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
    }}
    textarea {{ min-height: 150px; resize: vertical; line-height: 1.5; }}
    .row-list {{ display: flex; flex-direction: column; gap: 7px; margin-top: 12px; }}
    .row-link {{
      display: grid;
      grid-template-columns: 46px 1fr;
      gap: 8px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(23, 35, 61, 0.8);
    }}
    .row-link.done span {{ color: var(--good); }}
    .row-link.blank span {{ color: var(--warn); }}
    .row-link.active {{ border-color: var(--accent); background: rgba(29, 78, 216, 0.35); }}
    .row-link b {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }}
    .workspace {{ display: grid; grid-template-columns: minmax(360px, 48%) 1fr; gap: 20px; align-items: start; }}
    .card {{ background: rgba(17, 28, 51, 0.78); border: 1px solid var(--border); border-radius: 18px; padding: 16px; }}
    .image-wrap {{
      background: #050b16;
      border: 1px solid var(--border);
      border-radius: 14px;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .image-wrap img {{ max-width: 100%; max-height: 78vh; display: block; }}
    .topline {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 14px; }}
    .poster {{ font-size: 22px; font-weight: 800; letter-spacing: -0.03em; }}
    .nav {{ display: flex; gap: 10px; }}
    label {{ display: block; margin: 14px 0 7px; color: var(--muted); font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; }}
    .actions {{ display: flex; gap: 10px; align-items: center; margin-top: 14px; flex-wrap: wrap; }}
    button.primary {{ background: #2563eb; border-color: #3b82f6; font-weight: 800; }}
    button.secondary {{ background: var(--panel2); }}
    #status {{ min-height: 22px; }}
    .guidelines li {{ margin: 8px 0; color: var(--muted); }}
    code {{ background: #081121; border: 1px solid var(--border); padding: 2px 6px; border-radius: 6px; }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: none; }}
      .workspace {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">Free-form Prompt App</div>
      <div class="datasets">{''.join(nav_links)}</div>
      <div class="stats">
        <div class="stat"><b>{filled}</b><span class="muted small">filled</span></div>
        <div class="stat"><b>{blank}</b><span class="muted small">blank</span></div>
        <div class="stat"><b>{total}</b><span class="muted small">total</span></div>
      </div>
      <div class="progress"><div class="bar"></div></div>
      <div class="muted small">CSV: {html.escape(str(dataset.csv_path))}</div>
      <div class="muted small">Images: {html.escape(str(dataset.image_dir))}</div>

      <div class="filters">
        {filter_link('blank', 'Blank')}
        {filter_link('done', 'Done')}
        {filter_link('all', 'All')}
      </div>
      <form class="search" method="get" action="/dataset/{dataset.key}">
        <input type="hidden" name="filter" value="{html.escape(filter_name)}">
        <input name="q" placeholder="Search image/prompt..." value="{escaped_query}">
        <button type="submit">Search</button>
      </form>
      <div class="muted small">{len(indices)} matching rows</div>
      <div class="row-list">{''.join(row_links) if row_links else '<div class="muted">No rows match this filter.</div>'}</div>
    </aside>

    <main>
      <div class="topline">
        <div>
          <div class="muted small">{html.escape(dataset.label)} · row {(selected_index + 1) if selected_index is not None else 0} of {total}</div>
          <div class="poster">{escaped_poster or 'No row selected'}</div>
        </div>
        <div class="nav">
          <a class="navbtn" href="{previous_link}">← Previous</a>
          <a class="navbtn" id="nextLink" href="{next_link}">Next →</a>
        </div>
      </div>
      <div class="workspace">
        <section class="card">
          <div class="image-wrap">
            {'<img src="' + html.escape(image_url) + '" alt="' + escaped_poster + '">' if image_url else '<div class="muted">No image selected.</div>'}
          </div>
          <p class="muted small">Author prompts from the image/background only. Do not inspect GT boxes or annotation CSVs.</p>
        </section>
        <section class="card">
          <form id="promptForm">
            <input type="hidden" name="dataset" value="{html.escape(dataset.key)}">
            <input type="hidden" name="row_index" value="{selected_index if selected_index is not None else ''}">
            <label>Text prompt</label>
            <textarea name="text_prompt" id="textPrompt" placeholder="Example: Create a clean poster with one text box for the title at top-center, two text boxes at middle-center, and a small logo at bottom-right.">{escaped_prompt}</textarea>

            <label>Author ID</label>
            <input name="author_id" value="{escaped_author}" placeholder="e.g. designer1">

            <label>Independent of ground truth?</label>
            <select name="independent_of_ground_truth">
              <option value="yes" {'selected' if escaped_independent.lower() == 'yes' else ''}>yes</option>
              <option value="no" {'selected' if escaped_independent.lower() == 'no' else ''}>no</option>
            </select>

            <div class="actions">
              <button class="primary" type="submit">Save prompt</button>
              <button class="secondary" type="button" id="saveNext">Save + next</button>
              <span id="status" class="muted"></span>
            </div>
          </form>

          <div class="guidelines">
            <h3>Writing rules for {html.escape(dataset.label)}</h3>
            <ul>
              <li>One natural instruction, usually 12–35 words.</li>
              <li>Use only these model classes/synonyms for this dataset:</li>
              {class_guidance}
              <li>Use exact 3×3 spatial phrases when asking for position: <code>top-left</code>, <code>top-center</code>, <code>top-right</code>, <code>middle-left</code>, <code>middle-center</code>, <code>middle-right</code>, <code>bottom-left</code>, <code>bottom-center</code>, <code>bottom-right</code>.</li>
              {extra_guidance}
              <li>Do not copy any ground-truth layout. Write what a designer/user would request.</li>
            </ul>
          </div>
        </section>
      </div>
    </main>
  </div>
  <script>
    const form = document.getElementById('promptForm');
    const statusEl = document.getElementById('status');
    const nextLink = document.getElementById('nextLink');
    async function savePrompt(goNext=false) {{
      statusEl.textContent = 'Saving...';
      const response = await fetch('/api/save', {{
        method: 'POST',
        body: new URLSearchParams(new FormData(form))
      }});
      const data = await response.json();
      if (!response.ok || !data.ok) {{
        statusEl.textContent = data.error || 'Save failed';
        statusEl.style.color = 'var(--danger)';
        return;
      }}
      statusEl.textContent = 'Saved';
      statusEl.style.color = 'var(--good)';
      if (goNext && nextLink.getAttribute('href') !== '#') {{
        window.location.href = nextLink.getAttribute('href');
      }}
    }}
    form.addEventListener('submit', (event) => {{
      event.preventDefault();
      savePrompt(false);
    }});
    document.getElementById('saveNext').addEventListener('click', () => savePrompt(true));
    document.addEventListener('keydown', (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {{
        event.preventDefault();
        savePrompt(true);
      }}
    }});
  </script>
</body>
</html>"""


class PromptHandler(BaseHTTPRequestHandler):
    server_version = "PromptAuthoringHTTP/1.0"

    @property
    def app_config(self) -> AppConfig:
        return self.server.app_config  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[prompt-app] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        try:
            self.route_get()
        except Exception as exc:  # keep useful error visible in browser
            html_response(
                self,
                f"<h1>Prompt app error</h1><pre>{html.escape(type(exc).__name__ + ': ' + str(exc))}</pre>",
                500,
            )

    def do_POST(self) -> None:  # noqa: N802
        try:
            self.route_post()
        except Exception as exc:
            json_response(self, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)

    def route_get(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            first = next(iter(self.app_config.datasets))
            redirect(self, f"/dataset/{first}?filter=blank")
            return
        if path.startswith("/dataset/"):
            dataset_key = path.split("/", 2)[2]
            dataset = self.app_config.datasets.get(dataset_key)
            if not dataset:
                html_response(self, "Unknown dataset", 404)
                return
            params = parse_qs(parsed.query)
            filter_name = params.get("filter", ["blank"])[0]
            if filter_name not in {"blank", "done", "all"}:
                filter_name = "blank"
            query = params.get("q", [""])[0].strip()
            index_raw = params.get("index", [None])[0]
            selected_index = int(index_raw) if index_raw not in (None, "") else None
            html_response(self, render_page(self.app_config, dataset, selected_index, filter_name, query))
            return
        if path.startswith("/image/"):
            parts = path.split("/", 3)
            if len(parts) < 4:
                self.send_error(404)
                return
            dataset = self.app_config.datasets.get(parts[2])
            if not dataset:
                self.send_error(404)
                return
            image = safe_image_path(dataset, parts[3])
            if image is None or not image.exists() or not image.is_file():
                self.send_error(404, f"Image not found: {parts[3]}")
                return
            ctype = mimetypes.guess_type(str(image))[0] or "application/octet-stream"
            payload = image.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def route_post(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            json_response(self, {"ok": False, "error": "unknown endpoint"}, 404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body, keep_blank_values=True)
        dataset_key = form.get("dataset", [""])[0]
        dataset = self.app_config.datasets.get(dataset_key)
        if not dataset:
            json_response(self, {"ok": False, "error": "unknown dataset"}, 400)
            return
        row_index_raw = form.get("row_index", [""])[0]
        if not row_index_raw.isdigit():
            json_response(self, {"ok": False, "error": "invalid row_index"}, 400)
            return
        row_index = int(row_index_raw)
        text_prompt = form.get("text_prompt", [""])[0].strip()
        author_id = form.get("author_id", [""])[0].strip()
        independent = form.get("independent_of_ground_truth", ["yes"])[0].strip() or "yes"
        if independent not in {"yes", "no"}:
            independent = "yes"
        with CSV_LOCK:
            fieldnames, rows, prompt_column = read_rows(dataset)
            if not 0 <= row_index < len(rows):
                json_response(self, {"ok": False, "error": "row index outside CSV"}, 400)
                return
            if self.app_config.backup_on_save:
                make_backup(dataset)
            rows[row_index][prompt_column] = text_prompt
            rows[row_index]["author_id"] = author_id
            rows[row_index]["independent_of_ground_truth"] = independent
            atomic_write_csv(dataset, fieldnames, rows)
            filled = sum(not row_is_blank(row, prompt_column) for row in rows)
        json_response(
            self,
            {
                "ok": True,
                "row_index": row_index,
                "filled": filled,
                "total": len(rows),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("PROMPT_APP_CONFIG", DEFAULT_CONFIG)))
    parser.add_argument("--host", default=None, help="Override host from config.json")
    parser.add_argument("--port", type=int, default=None, help="Override port from config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    host = args.host or cfg.host
    port = args.port or cfg.port
    server = ThreadingHTTPServer((host, port), PromptHandler)
    server.app_config = cfg  # type: ignore[attr-defined]
    print(f"Prompt app running at http://{host}:{port}")
    print(f"Config: {args.config.resolve()}")
    for key, dataset in cfg.datasets.items():
        print(f"  {key}: csv={dataset.csv_path} images={dataset.image_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping prompt app.")


if __name__ == "__main__":
    main()
