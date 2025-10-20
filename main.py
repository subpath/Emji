import os

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
import sqlite3
import sqlite_vec
import typer
from typing import List
from pathlib import Path
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import struct
import pyperclip
from InquirerPy import inquirer
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    SpinnerColumn,
    TextColumn,
)
import multiprocessing
from rich.table import Table
from rich.console import Console
import urllib.request
import shutil

multiprocessing.set_start_method("spawn", force=True)

app = typer.Typer(add_completion=False, help="Emoji semantic search CLI 🔍")

HOME_DIR = Path.home() / ".emji"
HOME_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE = HOME_DIR / ".config"
DEFAULT_CONFIG = {
    "ALPHA": 0.2,  # ranking balance between your past clicks and cosine similarity
    "MODEL_URL": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_qint8_arm64.onnx",
    "EMOJI_URL": "https://gist.githubusercontent.com/subpath/13bd5c15f76f451dfcb85421a53f0666/raw/1d362e4b4addfcd920b88f949090c6e82bf2c791/emojies_shortnames.json",
}


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG
    with open(CONFIG_FILE) as f:
        return json.load(f)


CONFIG = load_config()

MODEL_FILE = HOME_DIR / "model_qint8_arm64.onnx"
DEFAULT_EMOJI_FILE = HOME_DIR / "shortnames.json"
OVERRIDE_EMOJI_FILE = HOME_DIR / "shortnames_override.json"
DB_FILE = HOME_DIR / "emoji_index.db"
ALPHA = CONFIG["ALPHA"]


def interactive_download(path: Path, url: str, label: str):
    consent = inquirer.select(
        message=f"{label} not found. Download it?",
        choices=["yes", "no"],
        default="yes",
    ).execute()
    if consent == "no":
        typer.echo(f"Cannot continue without {label}. Exiting.")
        raise typer.Exit(1)

    typer.echo(f"Downloading {label}...")
    with urllib.request.urlopen(url) as response, open(path, "wb") as out_file:
        total = int(response.getheader("Content-Length", 0))
        total_mb = total / (1024 * 1024)
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]{label}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed:.1f}/{task.total:.1f} MB"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Fetching {label}", total=total_mb)
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                progress.update(task, completed=downloaded / (1024 * 1024))
    typer.echo(f"Downloaded {label} → {path}")


def ensure_dependencies():
    if not MODEL_FILE.exists():
        interactive_download(MODEL_FILE, CONFIG["MODEL_URL"], "Model")
    if not DEFAULT_EMOJI_FILE.exists():
        interactive_download(DEFAULT_EMOJI_FILE, CONFIG["EMOJI_URL"], "Emoji data")


def get_emoji_file() -> Path:
    return OVERRIDE_EMOJI_FILE if OVERRIDE_EMOJI_FILE.exists() else DEFAULT_EMOJI_FILE


def encode(text: str) -> np.ndarray:
    inputs = tokenizer(
        text, return_tensors="np", padding="max_length", truncation=True, max_length=128
    )
    outputs = session.run(None, dict(inputs))
    last_hidden_state = outputs[0]
    mask = inputs["attention_mask"]
    emb = (last_hidden_state * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0].astype(np.float32)


def sanitize(s: str) -> str:
    return s.lower().replace("_", " ").replace(":", "").strip()


def connect_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_FILE)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def serialize_f32(vector: List[float]) -> bytes:
    return struct.pack("%sf" % len(vector), *vector)


def update_popularity(conn, query: str, emoji: str, event: str):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO popularity(query, emoji, shown, clicks)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(query, emoji)
        DO UPDATE SET
            shown = shown + (CASE WHEN ? = 'shown' THEN 1 ELSE 0 END),
            clicks = clicks + (CASE WHEN ? = 'click' THEN 1 ELSE 0 END)
        """,
        (
            query,
            emoji,
            1 if event == "shown" else 0,
            1 if event == "click" else 0,
            event,
            event,
        ),
    )
    conn.commit()


def build_index():
    ensure_dependencies()
    if DB_FILE.exists():
        typer.confirm(f"{DB_FILE} exists. Rebuild?", abort=True)
        DB_FILE.unlink()

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE emojis (
            id INTEGER PRIMARY KEY,
            name TEXT,
            sanitized TEXT,
            emoji TEXT
        )
    """
    )
    cur.execute(
        "CREATE VIRTUAL TABLE emoji_embeddings USING vec0(embedding float[384])"
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS popularity (
            query TEXT,
            emoji TEXT,
            shown INTEGER DEFAULT 0,
            clicks INTEGER DEFAULT 0,
            PRIMARY KEY (query, emoji)
        )
    """
    )

    with open(get_emoji_file()) as f:
        emojis = json.load(f)

    typer.echo("Building embeddings...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Embedding emojis", total=len(emojis))
        for name, symbol in emojis.items():
            sanitized = sanitize(name)
            emb = encode(sanitized)
            cur.execute(
                "INSERT INTO emojis (name, sanitized, emoji) VALUES (?, ?, ?)",
                (name, sanitized, symbol),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO emoji_embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, serialize_f32(emb)),
            )
            progress.update(task, advance=1)

    conn.commit()
    conn.close()
    typer.echo(f"✅ Index built with {len(emojis)} entries at {DB_FILE}")


def query_emoji(text: str, n: int = 3):
    ensure_dependencies()
    if not DB_FILE.exists():
        choice = inquirer.select(
            message="Emoji index not found. Build it now?",
            choices=["yes", "no"],
            default="yes",
        ).execute()
        if choice == "no":
            typer.echo("Exiting. Index required for search.")
            raise typer.Exit(1)
        build_index()

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT e.name, e.emoji, distance
        FROM emoji_embeddings v
        JOIN emojis e ON v.rowid = e.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        [serialize_f32(encode(sanitize(text))), n * 2],
    )
    results = cur.fetchall()

    cur.execute("SELECT emoji, clicks, shown FROM popularity WHERE query = ?", (text,))
    pop = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    def score(row, rank):
        clicks, shown = pop.get(row[1], (0, 0))
        cosine_sim = 1 / (1 + row[2])
        ctr = (clicks + 1) / (shown + 2)

        trust = max(0.0, min(1.0, (shown - 5) / 5))

        discount = 1 / np.log2(rank + 2)
        adjusted_ctr = ctr * (1 + discount)

        # Blend: CTR only influences ranking after enough impressions
        hybrid = (1 - trust) * cosine_sim + trust * (
            ALPHA * cosine_sim + (1 - ALPHA) * adjusted_ctr
        )
        return hybrid

    results = sorted(
        [(r, i) for i, r in enumerate(results)],
        key=lambda ri: score(ri[0], ri[1]),
        reverse=True,
    )[:n]

    results = [r for r, _ in results]

    for _, emoji, _ in results:
        update_popularity(conn, text, emoji, "shown")

    if not results:
        typer.echo("No results found.")
        conn.close()
        raise typer.Exit(0)

    choices = [
        f"{emoji} {name} {round(1 / (1 + distance), 3)}"
        for name, emoji, distance in results
    ]
    selected = inquirer.select(
        message="Select the correct emoji:", choices=choices, default=choices[0]
    ).execute()
    chosen_index = choices.index(selected)
    chosen_emoji = results[chosen_index][1]

    update_popularity(conn, text, chosen_emoji, "click")

    conn.close()
    pyperclip.copy(chosen_emoji)
    typer.echo(f"{chosen_emoji} copied to clipboard")


def show_stats(n: int = 10):
    if not DB_FILE.exists():
        typer.echo("No index found. Build it first.")
        raise typer.Exit(1)

    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT emoji, SUM(clicks) as total_clicks, SUM(shown) as total_shown
        FROM popularity
        GROUP BY emoji
        HAVING total_clicks != 0
        ORDER BY total_clicks DESC
        LIMIT ?
    """,
        (n,),
    )
    stats = cur.fetchall()

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Emoji", justify="center", style="bold")
    table.add_column("Clicks", justify="right")
    table.add_column("Shown", justify="right")
    table.add_column("Top 3 Queries", justify="left")

    for emoji, clicks, shown in stats:
        cur.execute(
            """
            SELECT query, clicks
            FROM popularity
            WHERE emoji = ?
            AND clicks != 0
            ORDER BY clicks DESC
            LIMIT 3
        """,
            (emoji,),
        )
        top_queries = [f"{q} ({c})" for q, c in cur.fetchall()]
        print(top_queries)
        table.add_row(emoji, str(clicks), str(shown), ", ".join(top_queries))

    conn.close()
    console.print(table)


@app.command()
def main(
    ctx: typer.Context = typer.Option(None, hidden=True),
    query: List[str] = typer.Argument(None, help="Text to search emojis for"),
    n: int = typer.Option(3, "--n", help="Number of emoji results to return"),
    build_index_flag: bool = typer.Option(
        False, "--build-index", help="Force rebuild the emoji index"
    ),
    cleanup_flag: bool = typer.Option(
        False, "--cleanup", help="Delete all Emji data and config (~/.emji)"
    ),
    show_stats_flag: bool = typer.Option(
        False, "--show-stats", help="Show emoji popularity statistics"
    ),
):
    if not query and not build_index_flag and not cleanup_flag and not show_stats_flag:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)

    if show_stats_flag:
        show_stats(n)
        raise typer.Exit(0)

    if cleanup_flag:
        confirm = inquirer.select(
            message=f"Delete all files in {HOME_DIR}? This cannot be undone.",
            choices=["yes", "no"],
            default="no",
        ).execute()
        if confirm == "yes":
            shutil.rmtree(HOME_DIR, ignore_errors=True)
            typer.echo(f"🧹 Removed {HOME_DIR}")
        else:
            typer.echo("Cleanup aborted.")
        raise typer.Exit(0)

    ensure_dependencies()

    if build_index_flag:
        build_index()
        return

    query_emoji(" ".join(query), n=n)


ensure_dependencies()
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
session = InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])

if __name__ == "__main__":
    app()
