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
import urllib.request

multiprocessing.set_start_method("spawn", force=True)

app = typer.Typer(add_completion=False, help="Emoji semantic search CLI 🔍")

MODEL_FILE = Path("model_qint8_arm64.onnx")
DEFAULT_EMOJI_FILE = Path("shortnames.json")
OVERRIDE_EMOJI_FILE = Path("shortnames_override.json")
DB_FILE = Path("emoji_index.db")

MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_qint8_arm64.onnx"
EMOJI_URL = "https://gist.githubusercontent.com/subpath/13bd5c15f76f451dfcb85421a53f0666/raw/1d362e4b4addfcd920b88f949090c6e82bf2c791/emojies_shortnames.json"


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
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[progress.description]{label}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed} of {task.total} bytes"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Fetching {label}", total=total)
            downloaded = 0
            chunk_size = 8192
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                progress.update(task, completed=downloaded)
    typer.echo(f"Downloaded {label} → {path}")


def ensure_dependencies():
    if not MODEL_FILE.exists():
        interactive_download(MODEL_FILE, MODEL_URL, "Model")
    if not DEFAULT_EMOJI_FILE.exists():
        interactive_download(DEFAULT_EMOJI_FILE, EMOJI_URL, "Emoji data")


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


def build_index():
    ensure_dependencies()
    if DB_FILE.exists():
        typer.confirm(f"{DB_FILE} exists. Rebuild?", abort=True)
        DB_FILE.unlink()

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE emojis (
            id INTEGER PRIMARY KEY,
            name TEXT,
            sanitized TEXT,
            emoji TEXT
        )
    """)
    cur.execute(
        "CREATE VIRTUAL TABLE emoji_embeddings USING vec0(embedding float[384])"
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
        [serialize_f32(encode(sanitize(text))), n],
    )
    results = cur.fetchall()
    conn.close()

    if not results:
        typer.echo("No results found.")
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
    pyperclip.copy(chosen_emoji)
    typer.echo(f"{chosen_emoji} copied to clipboard")


@app.command()
def main(
    ctx: typer.Context = typer.Option(None, hidden=True),
    query: List[str] = typer.Argument(None, help="Text to search emojis for"),
    n: int = typer.Option(3, "--n", help="Number of emoji results to return"),
    build_index_flag: bool = typer.Option(
        False, "--build-index", help="Force rebuild the emoji index"
    ),
):
    if not query and not build_index_flag:
        typer.echo(ctx.get_help())
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
