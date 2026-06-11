"""
playlist_engine.py – Path planning + song retrieval + playlist assembly.
"""

from __future__ import annotations

import math
import uuid
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd

from .mood_grid import MoodPoint, get_mood
from .models import (
    MoodInfo,
    PlaylistMetrics,
    PlaylistResponse,
    TrackInfo,
    WaypointInfo,
)
from .database import get_user_preferences, update_session_playlist, get_session, add_rejected_track


# ── Globals (loaded once at startup) ────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_song_df: Optional[pd.DataFrame] = None
_faiss_index: Optional[faiss.IndexFlatL2] = None


def load_artifacts() -> None:
    """Load FAISS index and song metadata into module-level globals."""
    global _song_df, _faiss_index

    parquet_path = DATA_DIR / "song_metadata.parquet"
    faiss_path = DATA_DIR / "song_index.faiss"

    if not parquet_path.exists() or not faiss_path.exists():
        raise FileNotFoundError(
            "Data artifacts not found. Run `python -m src.data_pipeline` first."
        )

    _song_df = pd.read_parquet(parquet_path)
    _faiss_index = faiss.read_index(str(faiss_path))
    print(f"Loaded {len(_song_df)} songs + FAISS index ({_faiss_index.ntotal} vectors)")


def _ensure_loaded():
    if _song_df is None or _faiss_index is None:
        load_artifacts()


def _quadrant(v: float, a: float) -> int:
    """Map (valence, arousal/energy) to circumplex quadrant 1-4."""
    if v >= 0.5 and a >= 0.5:
        return 1   # Happy / Excited
    if v < 0.5 and a >= 0.5:
        return 2   # Tense / Angry
    if v < 0.5 and a < 0.5:
        return 3   # Sad / Depressed
    return 4       # Calm / Relaxed


# ── 1. Path planning ───────────────────────────────────────────────────────

def plan_waypoints(
    start: Tuple[float, float],
    target: Tuple[float, float],
    n_waypoints: int = 5,
    path_type: str = "linear",
) -> List[Tuple[float, float]]:
    """Generate *n_waypoints* coordinates according to path_type (linear, creative, random)."""
    sv, sa = start
    tv, ta = target
    if n_waypoints < 2:
        return [start, target]

    if path_type == "creative":
        # Quadratic Bezier Curve with a perpendicular control point
        dv = tv - sv
        da = ta - sa
        dist = math.sqrt(dv * dv + da * da)
        
        if dist < 0.1:
            return [
                (sv + i / (n_waypoints - 1) * dv, sa + i / (n_waypoints - 1) * da)
                for i in range(n_waypoints)
            ]
            
        mv = sv + dv / 2.0
        ma = sa + da / 2.0
        
        # Perpendicular direction
        pv = -da / dist
        pa = dv / dist
        
        # Shift control point by 0.25 to make it an elegant arc
        ctrl_v = mv + 0.25 * pv
        ctrl_a = ma + 0.25 * pa
        
        # Clamp control point
        ctrl_v = max(0.0, min(1.0, ctrl_v))
        ctrl_a = max(0.0, min(1.0, ctrl_a))
        
        waypoints = []
        for i in range(n_waypoints):
            t = i / (n_waypoints - 1)
            wv = (1 - t)**2 * sv + 2 * (1 - t) * t * ctrl_v + t**2 * tv
            wa = (1 - t)**2 * sa + 2 * (1 - t) * t * ctrl_a + t**2 * ta
            waypoints.append((float(wv), float(wa)))
        return waypoints

    elif path_type == "random":
        # Linear path + Gaussian noise
        waypoints = []
        for i in range(n_waypoints):
            t = i / (n_waypoints - 1)
            wv = sv + t * (tv - sv)
            wa = sa + t * (ta - sa)
            if 0 < i < n_waypoints - 1:
                wv += random.gauss(0.0, 0.05)
                wa += random.gauss(0.0, 0.05)
            wv = max(0.0, min(1.0, wv))
            wa = max(0.0, min(1.0, wa))
            waypoints.append((float(wv), float(wa)))
        return waypoints

    else:  # "linear"
        return [
            (sv + i / (n_waypoints - 1) * (tv - sv),
             sa + i / (n_waypoints - 1) * (ta - sa))
            for i in range(n_waypoints)
        ]


# ── 2. Song retrieval ──────────────────────────────────────────────────────

def _retrieve_candidates(
    waypoint: Tuple[float, float],
    k: int = 20,
    genre_filter: Optional[List[str]] = None,
    v_bias: float = 0.0,
    a_bias: float = 0.0,
) -> pd.DataFrame:
    """Return top-k candidate songs nearest to *waypoint* in VA space, adjusted by user bias."""
    _ensure_loaded()
    
    # Adjust query coordinates based on user personalization bias
    query_v = max(0.0, min(1.0, waypoint[0] - v_bias))
    query_a = max(0.0, min(1.0, waypoint[1] - a_bias))

    query = np.array([[query_v, query_a]], dtype=np.float32)
    search_k = k * 5 if genre_filter else k
    distances, indices = _faiss_index.search(query, search_k)

    cands = _song_df.iloc[indices[0]].copy()
    cands["va_distance"] = distances[0]

    if genre_filter:
        lowered = [g.lower() for g in genre_filter]
        cands = cands[cands["track_genre"].str.lower().isin(lowered)]

    return cands.head(k).reset_index(drop=True)


def _score_candidates(
    cands: pd.DataFrame,
    waypoint: Tuple[float, float],
    prev_tempo_norm: Optional[float],
    w_proximity: float = 0.55,
    w_tempo: float = 0.20,
    w_popularity: float = 0.25,
) -> pd.DataFrame:
    """Compute a composite score for each candidate song."""
    max_dist = math.sqrt(2)  # diagonal of [0,1]²
    cands["score_proximity"] = 1.0 - (np.sqrt(cands["va_distance"]) / max_dist)

    # Tempo continuity score
    if prev_tempo_norm is not None and "tempo_norm" in cands.columns:
        cands["score_tempo"] = 1.0 - (cands["tempo_norm"] - prev_tempo_norm).abs()
    else:
        cands["score_tempo"] = 0.5

    # Popularity score (normalised 0-1)
    if "popularity" in cands.columns:
        cands["score_pop"] = cands["popularity"] / 100.0
    else:
        cands["score_pop"] = 0.5

    cands["composite_score"] = (
        w_proximity * cands["score_proximity"]
        + w_tempo * cands["score_tempo"]
        + w_popularity * cands["score_pop"]
    )
    return cands.sort_values("composite_score", ascending=False)


# ── 3. Playlist assembly ───────────────────────────────────────────────────

def _assemble_playlist(
    waypoints: List[Tuple[float, float]],
    songs_per_waypoint: int = 3,
    genre_filter: Optional[List[str]] = None,
    max_per_artist: int = 2,
    v_bias: float = 0.0,
    a_bias: float = 0.0,
    exclude_ids: Optional[set] = None,
) -> Tuple[List[dict], float]:
    """Select songs along the waypoint path, returning (tracks, last_tempo)."""
    selected_ids = set() if exclude_ids is None else set(exclude_ids)
    artist_counts = Counter()
    tracks = []
    prev_tempo = None

    for wi, wp in enumerate(waypoints):
        cands = _retrieve_candidates(wp, k=30, genre_filter=genre_filter, v_bias=v_bias, a_bias=a_bias)
        scored = _score_candidates(cands, wp, prev_tempo)

        picked = 0
        for _, row in scored.iterrows():
            if picked >= songs_per_waypoint:
                break
            tid = row["track_id"]
            artist = row["artists"]
            
            if tid in selected_ids:
                continue
            if artist_counts[artist] >= max_per_artist:
                continue

            selected_ids.add(tid)
            artist_counts[artist] += 1
            prev_tempo = row.get("tempo_norm", 0.5)

            tracks.append({
                "position": len(tracks) + 1,
                "track_id": tid,
                "track_name": row["track_name"],
                "artists": artist,
                "album_name": row.get("album_name", ""),
                "valence": float(row["valence"]),
                "arousal": float(row["energy"]),
                "genre": row.get("track_genre", "unknown"),
                "waypoint_index": wi,
                "spotify_url": f"https://open.spotify.com/track/{tid}",
            })
            picked += 1

    return tracks, prev_tempo or 0.5


# ── 4. Quality metrics (Study Section 10.1) ────────────────────────────────

def _compute_metrics(
    tracks: List[dict],
    start: Tuple[float, float],
    target: Tuple[float, float],
    iso_threshold: float = 0.20,
) -> PlaylistMetrics:
    """Compute automated quality metrics for the generated playlist."""
    if not tracks:
        return PlaylistMetrics(
            endpoint_accuracy=1.0,
            smoothness_score=1.0,
            max_jump=1.0,
            iso_compliance=False,
            genre_entropy=0.0,
        )

    coords = np.array([[t["valence"], t["arousal"]] for t in tracks])
    target_arr = np.array(target)
    start_arr = np.array(start)

    endpoint_acc = float(np.linalg.norm(coords[-1] - target_arr))

    if len(coords) > 1:
        diffs = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        smoothness = float(diffs.mean())
        max_jump = float(diffs.max())
    else:
        smoothness = 0.0
        max_jump = 0.0

    iso = float(np.linalg.norm(coords[0] - start_arr)) < iso_threshold

    genres = [t["genre"] for t in tracks]
    genre_counts = Counter(genres)
    total = len(genres)
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in genre_counts.values()
        if c > 0
    )

    return PlaylistMetrics(
        endpoint_accuracy=round(endpoint_acc, 4),
        smoothness_score=round(smoothness, 4),
        max_jump=round(max_jump, 4),
        iso_compliance=iso,
        genre_entropy=round(entropy, 4),
    )


# ── 5. Public entry points ──────────────────────────────────────────────────

def generate_playlist(
    start_valence: float,
    start_arousal: float,
    target_valence: float,
    target_arousal: float,
    n_songs: int = 15,
    genre_filter: Optional[List[str]] = None,
    user_id: str = "default_user",
    path_type: str = "linear",
) -> PlaylistResponse:
    """Full pipeline: continuous coordinates → waypoints → retrieval → assembly → response."""
    _ensure_loaded()

    # Get user preferences
    prefs = get_user_preferences(user_id)
    v_bias = prefs.get("valence_bias", 0.0)
    a_bias = prefs.get("arousal_bias", 0.0)

    start = (start_valence, start_arousal)
    target = (target_valence, target_arousal)

    n_waypoints = max(3, n_songs // 3)
    songs_per_wp = max(1, n_songs // n_waypoints)

    waypoints = plan_waypoints(start, target, n_waypoints, path_type)
    tracks, _ = _assemble_playlist(
        waypoints,
        songs_per_waypoint=songs_per_wp,
        genre_filter=genre_filter,
        max_per_artist=2,
        v_bias=v_bias,
        a_bias=a_bias,
    )

    tracks = tracks[:n_songs]
    for i, t in enumerate(tracks):
        t["position"] = i + 1

    metrics = _compute_metrics(tracks, start, target)
    session_id = str(uuid.uuid4())

    return PlaylistResponse(
        session_id=session_id,
        mood_start=MoodInfo(
            label="Custom Start", emoji="🏁",
            valence=start_valence, arousal=start_arousal,
            quadrant=_quadrant(start_valence, start_arousal), color="#818cf8",
        ),
        mood_target=MoodInfo(
            label="Custom Target", emoji="🎯",
            valence=target_valence, arousal=target_arousal,
            quadrant=_quadrant(target_valence, target_arousal), color="#00BFA5",
        ),
        waypoints=[
            WaypointInfo(index=i, valence=round(v, 4), arousal=round(a, 4))
            for i, (v, a) in enumerate(waypoints)
        ],
        playlist=[TrackInfo(**t) for t in tracks],
        metrics=metrics,
    )


def reroute_track(session_id: str, position: int, user_id: str = "default_user") -> dict:
    """Replace a single track at *position* (1-indexed) with an alternative.
    
    Enforces session-level exclusion: the replaced track_id is permanently
    added to rejected_track_ids so it won't re-appear in future reroutes.
    """
    import json
    
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found.")
        
    playlist = json.loads(session["playlist"])
    if position < 1 or position > len(playlist):
        raise ValueError(f"Invalid position {position}. Playlist has length {len(playlist)}.")
        
    target_track = playlist[position - 1]
    wi = target_track["waypoint_index"]
    
    # Persist the rejected track before replacing it
    add_rejected_track(session_id, target_track["track_id"])

    start = (session["start_valence"], session["start_arousal"])
    target = (session["target_valence"], session["target_arousal"])
    
    n_songs = len(playlist)
    n_waypoints = max(3, n_songs // 3)
    
    waypoints = plan_waypoints(start, target, n_waypoints, session["path_type"])
    wp = waypoints[wi]
    
    prefs = get_user_preferences(user_id)
    v_bias = prefs.get("valence_bias", 0.0)
    a_bias = prefs.get("arousal_bias", 0.0)
    
    # Session-level exclusion: all current tracks + all previously rejected IDs
    import json as _json
    session_rejected = set(_json.loads(session.get("rejected_track_ids") or "[]"))
    exclude_ids = {t["track_id"] for t in playlist} | session_rejected
    
    # Query candidates with user bias applied
    cands = _retrieve_candidates(wp, k=50, v_bias=v_bias, a_bias=a_bias)
    
    prev_tempo = None
    if position > 1:
        prev_track = playlist[position - 2]
        _ensure_loaded()
        prev_row = _song_df[_song_df["track_id"] == prev_track["track_id"]]
        if not prev_row.empty:
            prev_tempo = prev_row.iloc[0].get("tempo_norm", 0.5)
            
    scored = _score_candidates(cands, wp, prev_tempo)
    
    new_track = None
    for _, row in scored.iterrows():
        tid = row["track_id"]
        if tid in exclude_ids:
            continue
        
        new_track = {
            "position": position,
            "track_id": tid,
            "track_name": row["track_name"],
            "artists": row["artists"],
            "album_name": row.get("album_name", ""),
            "valence": float(row["valence"]),
            "arousal": float(row["energy"]),
            "genre": row.get("track_genre", "unknown"),
            "waypoint_index": wi,
            "spotify_url": f"https://open.spotify.com/track/{tid}",
        }
        break
        
    if not new_track:
        # Relaxed fallback: only skip the current track_id
        for _, row in scored.iterrows():
            tid = row["track_id"]
            if tid == target_track["track_id"]:
                continue
            new_track = {
                "position": position,
                "track_id": tid,
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album_name": row.get("album_name", ""),
                "valence": float(row["valence"]),
                "arousal": float(row["energy"]),
                "genre": row.get("track_genre", "unknown"),
                "waypoint_index": wi,
                "spotify_url": f"https://open.spotify.com/track/{tid}",
            }
            break
            
    if not new_track:
        raise ValueError("No alternative candidates found in database.")
        
    playlist[position - 1] = new_track
    new_metrics = _compute_metrics(playlist, start, target)
    
    update_session_playlist(session_id, json.dumps(playlist), new_metrics.model_dump_json())
    
    return {
        "track": new_track,
        "metrics": new_metrics.model_dump()
    }


def update_segment_waypoints(
    session_id: str,
    new_waypoints: List[Tuple[float, float]],
    user_id: str = "default_user",
) -> dict:
    """Regenerate playlist using a new set of waypoints while preserving
    session-level rejected IDs."""
    import json
    
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found.")

    prefs = get_user_preferences(user_id)
    v_bias = prefs.get("valence_bias", 0.0)
    a_bias = prefs.get("arousal_bias", 0.0)

    session_rejected = set(json.loads(session.get("rejected_track_ids") or "[]"))

    n_songs = len(json.loads(session["playlist"]))
    songs_per_wp = max(1, n_songs // len(new_waypoints))

    start = (session["start_valence"], session["start_arousal"])
    target = (session["target_valence"], session["target_arousal"])

    tracks, _ = _assemble_playlist(
        new_waypoints,
        songs_per_waypoint=songs_per_wp,
        max_per_artist=2,
        v_bias=v_bias,
        a_bias=a_bias,
        exclude_ids=session_rejected,
    )

    tracks = tracks[:n_songs]
    for i, t in enumerate(tracks):
        t["position"] = i + 1

    new_metrics = _compute_metrics(tracks, start, target)
    update_session_playlist(session_id, json.dumps(tracks), new_metrics.model_dump_json())

    return {
        "playlist": tracks,
        "waypoints": [
            {"index": i, "valence": round(v, 4), "arousal": round(a, 4)}
            for i, (v, a) in enumerate(new_waypoints)
        ],
        "metrics": new_metrics.model_dump(),
    }
