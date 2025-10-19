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

multiprocessing.set_start_method("spawn", force=True)

app = typer.Typer(help="Emoji semantic search CLI 🔍")

MODEL_FILE = Path("model_qint8_arm64.onnx")
EMOJI_FILE = Path("shortnames.json")
DB_FILE = Path("emoji_index.db")

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
session = InferenceSession(str(MODEL_FILE), providers=["CPUExecutionProvider"])


def encode(text: str) -> np.ndarray:
    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
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


@app.command("build-index")
def build_index():
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

    with open(EMOJI_FILE) as f:
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


@app.command("query")
def query_emoji(text: str, n: int = typer.Option(3, help="Number of emojis to return")):
    if not DB_FILE.exists():
        typer.echo("Index not found. Run: emoji build-index")
        raise typer.Exit(1)

    conn = connect_db()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT e.name, e.emoji, distance
        FROM emoji_embeddings v
        JOIN emojis e ON v.rowid = e.id
        WHERE v.embedding MATCH ?
              AND k = ?
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
        f"{emoji}  {name} {round(1 / (1 + distance), 3)}"
        for name, emoji, distance in results
    ]
    selected = inquirer.select(
        message="Select the correct emoji:", choices=choices, default=choices[0]
    ).execute()

    chosen_index = choices.index(selected)
    chosen_emoji = results[chosen_index][1]
    pyperclip.copy(chosen_emoji)
    typer.echo(f"{chosen_emoji} copied to clipboard")


if __name__ == "__main__":
    app()
