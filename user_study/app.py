"""
IntentDiT user study (Flask + SQLite).

Part A - protocol-matched pairwise preference
    Quality trials compare image-only IntentDiT with LayoutDiT. Instruction trials
    compare text-conditioned IntentDiT with a genuine text-conditioned baseline.
    Side order is randomized and ties are allowed.

Part B - severity rating
    The participant sees 5 prepared layout-failure exemplars per category:
        subject_occlusion, illegible_text, element_overlap,
        underlay_misalignment, count_mismatch.
    They rate severity on a 1-7 scale. This validates the paper's claim
    that Occlusion and Readability are the most design-critical metrics.

Storage
    SQLite database at user_study/data/study.sqlite
    Each response is a row tagged with participant_id, criterion, image_id,
    rendered_paths (a/b/c), choice, latency, free-form rationale.

Aggregation
    See user_study/aggregate_results.py and hierarchical_bootstrap.py.

Usage
-----
1. Render layouts (one per method) into user_study/data/renders/<method>/<image>.png
   using user_study/render_for_study.py.
2. Build the comparison manifest:
       python user_study/build_manifest.py --n 30 --seed 42
3. Launch:
       FLASK_ENV=production python user_study/app.py --port 5000

The app supports Prolific completion-code redirect via ?prolific_pid= query parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from flask import (Flask, abort, g, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DB = DATA / "study.sqlite"
MANIFEST = DATA / "manifest.json"

PART_A_SIZE = 60
PART_B_CATEGORIES = ["subject_occlusion", "illegible_text",
                     "element_overlap", "underlay_misalignment", "count_mismatch"]
PART_B_PER_CATEGORY = 5

app = Flask(__name__,
            template_folder=str(ROOT / "templates"),
            static_folder=str(ROOT / "static"))
app.secret_key = os.environ.get("INTENTDIT_USER_STUDY_KEY", "dev-secret-change-me")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        DATA.mkdir(parents=True, exist_ok=True)
        g.db = sqlite3.connect(str(DB))
        g.db.row_factory = sqlite3.Row
        g.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS participants (
                id TEXT PRIMARY KEY,
                designer INTEGER NOT NULL,
                prolific_pid TEXT,
                created_iso TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS responses_a (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant TEXT NOT NULL,
                trial INTEGER NOT NULL,
                image_id TEXT NOT NULL,
                left_method TEXT NOT NULL,
                right_method TEXT NOT NULL,
                criterion TEXT NOT NULL DEFAULT 'quality',
                prompt TEXT,
                choice TEXT NOT NULL,        -- 'left' | 'right' | 'tie'
                latency_ms INTEGER,
                rationale TEXT,
                created_iso TEXT NOT NULL,
                FOREIGN KEY(participant) REFERENCES participants(id)
            );
            CREATE TABLE IF NOT EXISTS responses_b (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant TEXT NOT NULL,
                category TEXT NOT NULL,
                example_id TEXT NOT NULL,
                severity INTEGER NOT NULL,
                latency_ms INTEGER,
                created_iso TEXT NOT NULL,
                FOREIGN KEY(participant) REFERENCES participants(id)
            );
            """
        )
        columns = {
            row["name"] for row in g.db.execute("PRAGMA table_info(responses_a)").fetchall()
        }
        if "criterion" not in columns:
            g.db.execute("ALTER TABLE responses_a ADD COLUMN criterion TEXT NOT NULL DEFAULT 'quality'")
        if "prompt" not in columns:
            g.db.execute("ALTER TABLE responses_a ADD COLUMN prompt TEXT")
        g.db.commit()
    return g.db


@app.teardown_appcontext
def close_db(_exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        return {"part_a": [], "part_b": []}
    with open(MANIFEST) as f:
        return json.load(f)


@app.route("/")
def index():
    manifest = load_manifest()
    n_a = len(manifest.get("part_a") or [])
    n_b = len(manifest.get("part_b") or [])
    return render_template("index.html",
                           part_a_size=PART_A_SIZE,
                           part_b_size=len(PART_B_CATEGORIES) * PART_B_PER_CATEGORY,
                           manifest_n_a=n_a,
                           manifest_n_b=n_b)


@app.route("/start", methods=["POST"])
def start():
    designer = 1 if request.form.get("designer") == "yes" else 0
    prolific_pid = request.args.get("prolific_pid")
    pid = f"P{int(time.time() * 1000)}{random.randint(0, 999):03d}"
    db = get_db()
    db.execute(
        "INSERT INTO participants (id, designer, prolific_pid, created_iso) "
        "VALUES (?, ?, ?, datetime('now'))",
        (pid, designer, prolific_pid),
    )
    db.commit()
    session["participant"] = pid
    session["designer"] = designer
    session["a_idx"] = 0
    session["b_idx"] = 0
    session["t0"] = time.time()
    return redirect(url_for("part_a"))


def _require_participant():
    pid = session.get("participant")
    if not pid:
        abort(403, "session expired - please return to / and start over.")
    return pid


@app.route("/part_a", methods=["GET"])
def part_a():
    pid = _require_participant()
    manifest = load_manifest()
    items = manifest["part_a"]
    if not items:
        return render_template("setup_incomplete.html",
                               reason="part_a",
                               pid=pid)
    idx = session.get("a_idx", 0)
    if idx >= min(PART_A_SIZE, len(items)):
        return redirect(url_for("part_b"))
    item = items[idx]
    return render_template("part_a.html",
                           item=item, idx=idx + 1, total=PART_A_SIZE,
                           pid=pid)


@app.route("/part_a", methods=["POST"])
def submit_a():
    pid = _require_participant()
    idx = session.get("a_idx", 0)
    db = get_db()
    db.execute(
        "INSERT INTO responses_a (participant, trial, image_id, left_method, right_method, "
        "criterion, prompt, choice, latency_ms, rationale, created_iso) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (
            pid, idx,
            request.form["image_id"], request.form["left_method"], request.form["right_method"],
            request.form["criterion"], request.form.get("prompt", ""),
            request.form["choice"],
            int(request.form.get("latency_ms", 0) or 0),
            request.form.get("rationale", ""),
        ),
    )
    db.commit()
    session["a_idx"] = idx + 1
    return redirect(url_for("part_a"))


@app.route("/part_b", methods=["GET"])
def part_b():
    pid = _require_participant()
    manifest = load_manifest()
    items = manifest["part_b"]
    if not items:
        return render_template("setup_incomplete.html",
                               reason="part_b",
                               pid=pid)
    idx = session.get("b_idx", 0)
    total = len(PART_B_CATEGORIES) * PART_B_PER_CATEGORY
    if idx >= min(total, len(items)):
        return redirect(url_for("done"))
    item = items[idx]
    return render_template("part_b.html",
                           item=item, idx=idx + 1, total=total, pid=pid)


@app.route("/part_b", methods=["POST"])
def submit_b():
    pid = _require_participant()
    idx = session.get("b_idx", 0)
    db = get_db()
    db.execute(
        "INSERT INTO responses_b (participant, category, example_id, severity, latency_ms, created_iso) "
        "VALUES (?, ?, ?, ?, ?, datetime('now'))",
        (
            pid,
            request.form["category"], request.form["example_id"],
            int(request.form["severity"]),
            int(request.form.get("latency_ms", 0) or 0),
        ),
    )
    db.commit()
    session["b_idx"] = idx + 1
    return redirect(url_for("part_b"))


@app.route("/done")
def done():
    pid = session.get("participant", "")
    return render_template("done.html", pid=pid,
                           prolific_url=os.environ.get("PROLIFIC_COMPLETION_URL"))


@app.route("/renders/<path:p>")
def serve_render(p):
    # Pairwise stimuli: data/renders/<method>/<file>.png
    # Part B failures: manifest uses paths like failures/<cat>/<file>.png under data/failures/
    if p.startswith("failures/"):
        return send_from_directory(DATA, p)
    return send_from_directory(DATA / "renders", p)


@app.route("/health")
def health():
    return jsonify({"ok": True, "n_participants": _row_count("participants"),
                    "n_a": _row_count("responses_a"), "n_b": _row_count("responses_b")})


def _row_count(table: str) -> int:
    db = get_db()
    return db.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
