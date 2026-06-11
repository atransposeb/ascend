"""
data_pipeline.py – Load CSV, clean, normalise, build FAISS index.

Run as a script:
    python -m src.data_pipeline
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


# ── Paths ───────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent          # emotional_journey/
DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = ROOT_DIR.parent / "dataset.csv"             # ascend/dataset.csv


# ── Feature columns we keep ─────────────────────────────────────────────────

KEEP_COLS = [
    "track_id", "artists", "album_name", "track_name",
    "popularity", "duration_ms", "explicit",
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "time_signature",
    "track_genre",
]


def _quadrant(v: float, a: float) -> int:
    """Map (valence, arousal/energy) to circumplex quadrant 1-4."""
    if v >= 0.5 and a >= 0.5:
        return 1   # Happy / Excited
    if v < 0.5 and a >= 0.5:
        return 2   # Tense / Angry
    if v < 0.5 and a < 0.5:
        return 3   # Sad / Depressed
    return 4       # Calm / Relaxed


def load_and_clean(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a cleaned DataFrame."""
    print(f"Loading dataset from {path} ...")
    df = pd.read_csv(path)

    # Drop the unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    initial_rows = len(df)

    # 1. Keep only the columns we need
    df = df[[c for c in KEEP_COLS if c in df.columns]].copy()

    # 2. Drop rows missing critical fields
    df = df.dropna(subset=["track_id", "valence", "energy", "track_name", "artists"])

    # 3. Remove duplicates on track_id – keep highest popularity
    df = df.sort_values("popularity", ascending=False).drop_duplicates(subset="track_id", keep="first")

    # 4. Remove very short tracks (< 30 s)
    if "duration_ms" in df.columns:
        df = df[df["duration_ms"] >= 30_000]

    # 5. Clamp valence & energy to [0, 1]
    df["valence"] = df["valence"].clip(0.0, 1.0)
    df["energy"] = df["energy"].clip(0.0, 1.0)

    # 6. Normalise tempo to [0, 1] via min-max
    if "tempo" in df.columns:
        t_min, t_max = df["tempo"].min(), df["tempo"].max()
        if t_max > t_min:
            df["tempo_norm"] = (df["tempo"] - t_min) / (t_max - t_min)
        else:
            df["tempo_norm"] = 0.5

    # 7. Normalise loudness to [0, 1]
    if "loudness" in df.columns:
        l_min, l_max = df["loudness"].min(), df["loudness"].max()
        if l_max > l_min:
            df["loudness_norm"] = (df["loudness"] - l_min) / (l_max - l_min)
        else:
            df["loudness_norm"] = 0.5

    # 8. Assign mood quadrant
    df["mood_quadrant"] = df.apply(lambda r: _quadrant(r["valence"], r["energy"]), axis=1)

    df = df.reset_index(drop=True)
    print(f"Cleaned: {initial_rows} → {len(df)} rows  "
          f"(dropped {initial_rows - len(df)})")
    return df


def build_faiss_index(df: pd.DataFrame) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index on the 2-D (valence, energy) vectors."""
    va = df[["valence", "energy"]].values.astype(np.float32)
    index = faiss.IndexFlatL2(2)
    index.add(va)
    print(f"FAISS index built – {index.ntotal} vectors, dim={index.d}")
    return index


def save_artifacts(df: pd.DataFrame, index: faiss.IndexFlatL2) -> dict:
    """Persist the cleaned metadata and FAISS index to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = DATA_DIR / "song_metadata.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved metadata → {parquet_path}")

    faiss_path = DATA_DIR / "song_index.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index → {faiss_path}")

    # Pipeline report
    report = {
        "total_tracks": len(df),
        "unique_artists": int(df["artists"].nunique()),
        "unique_genres": int(df["track_genre"].nunique()) if "track_genre" in df.columns else 0,
        "valence_range": [float(df["valence"].min()), float(df["valence"].max())],
        "energy_range": [float(df["energy"].min()), float(df["energy"].max())],
        "quadrant_distribution": df["mood_quadrant"].value_counts().to_dict(),
    }
    report_path = DATA_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved pipeline report → {report_path}")
    return report


# ── Entry-point ─────────────────────────────────────────────────────────────

def run_pipeline(csv_path: Path | str | None = None) -> dict:
    """Execute the full pipeline and return the report dict."""
    path = Path(csv_path) if csv_path else DATASET_PATH
    if not path.exists():
        print(f"ERROR: Dataset not found at {path}", file=sys.stderr)
        sys.exit(1)

    df = load_and_clean(path)
    index = build_faiss_index(df)
    report = save_artifacts(df, index)

    print("\n✅  Pipeline complete.")
    for k, v in report.items():
        print(f"   {k}: {v}")
    return report


if __name__ == "__main__":
    run_pipeline()
