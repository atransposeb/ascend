"""
mood_grid.py – 16-mood grid definition mapped to Russell's Circumplex Model.

Each mood is a named point in 2-D Valence-Arousal (VA) space [0, 1]².
Quadrants:
  Q1 (high V, high A) – Happy / Excited
  Q2 (low  V, high A) – Tense / Angry
  Q3 (low  V, low  A) – Sad / Depressed
  Q4 (high V, low  A) – Calm / Relaxed
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass(frozen=True)
class MoodPoint:
    """A single labelled point in VA space."""
    label: str
    emoji: str
    valence: float   # 0 = most negative, 1 = most positive
    arousal: float   # 0 = most calm,     1 = most energised
    quadrant: int    # 1-4 following Russell convention
    color: str       # hex colour for UI gradient seed

    def to_dict(self) -> dict:
        return asdict(self)


# ── 4×4 Grid (rows = arousal descending, cols = valence ascending) ──────────

MOOD_GRID: List[List[MoodPoint]] = [
    # Row 0 – HIGH AROUSAL
    [
        MoodPoint("Angry",      "😡", 0.15, 0.90, 2, "#E53935"),
        MoodPoint("Anxious",    "😰", 0.35, 0.85, 2, "#D81B60"),
        MoodPoint("Excited",    "🤩", 0.70, 0.90, 1, "#FB8C00"),
        MoodPoint("Euphoric",   "🎉", 0.90, 0.95, 1, "#FFD600"),
    ],
    # Row 1 – MID-HIGH AROUSAL
    [
        MoodPoint("Frustrated", "😤", 0.20, 0.65, 2, "#C62828"),
        MoodPoint("Tense",      "😬", 0.35, 0.60, 2, "#AD1457"),
        MoodPoint("Energized",  "⚡", 0.70, 0.65, 1, "#EF6C00"),
        MoodPoint("Happy",      "😊", 0.85, 0.70, 1, "#F9A825"),
    ],
    # Row 2 – MID-LOW AROUSAL
    [
        MoodPoint("Depressed",  "😞", 0.15, 0.35, 3, "#1565C0"),
        MoodPoint("Melancholic","🥀", 0.30, 0.40, 3, "#4527A0"),
        MoodPoint("Content",    "😌", 0.70, 0.35, 4, "#2E7D32"),
        MoodPoint("Peaceful",   "🕊️", 0.85, 0.40, 4, "#00897B"),
    ],
    # Row 3 – LOW AROUSAL
    [
        MoodPoint("Drained",    "😶", 0.10, 0.15, 3, "#283593"),
        MoodPoint("Sad",        "😢", 0.30, 0.20, 3, "#5C6BC0"),
        MoodPoint("Relaxed",    "😎", 0.65, 0.20, 4, "#43A047"),
        MoodPoint("Calm",       "🧘", 0.85, 0.15, 4, "#26A69A"),
    ],
]

# Flat lookup by label (case-insensitive)
MOOD_LOOKUP: Dict[str, MoodPoint] = {
    mood.label.lower(): mood
    for row in MOOD_GRID
    for mood in row
}

# Quadrant colour gradients for the frontend VA scatter plot
QUADRANT_COLORS = {
    1: {"from": "#FFD600", "to": "#FF6D00", "label": "Happy / Excited"},
    2: {"from": "#E53935", "to": "#AD1457", "label": "Tense / Angry"},
    3: {"from": "#1565C0", "to": "#4527A0", "label": "Sad / Low Energy"},
    4: {"from": "#00BFA5", "to": "#2E7D32", "label": "Calm / Relaxed"},
}


def get_mood(label: str) -> Optional[MoodPoint]:
    """Retrieve a MoodPoint by case-insensitive label."""
    return MOOD_LOOKUP.get(label.strip().lower())


def mood_grid_json() -> dict:
    """Return the full grid + quadrant info as a JSON-serialisable dict."""
    return {
        "grid": [[m.to_dict() for m in row] for row in MOOD_GRID],
        "quadrants": QUADRANT_COLORS,
    }
