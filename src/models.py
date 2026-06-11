"""
models.py – Pydantic schemas for API request/response bodies.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────────────────────────

class PlaylistRequest(BaseModel):
    start_valence: float = Field(..., ge=0.0, le=1.0, description="Starting valence (0.0 to 1.0)")
    start_arousal: float = Field(..., ge=0.0, le=1.0, description="Starting arousal/energy (0.0 to 1.0)")
    target_valence: float = Field(..., ge=0.0, le=1.0, description="Target valence (0.0 to 1.0)")
    target_arousal: float = Field(..., ge=0.0, le=1.0, description="Target arousal/energy (0.0 to 1.0)")
    n_songs: int = Field(15, ge=5, le=30, description="Number of songs in the playlist")
    genre_filter: Optional[List[str]] = Field(None, description="Optional list of genres to restrict to")
    user_id: str = Field("default_user", description="Identifier for personalization")
    path_type: str = Field("linear", description="Type of transition path ('linear', 'creative', 'random')")


class FeedbackRequest(BaseModel):
    session_id: str
    pre_valence: int = Field(..., ge=1, le=9, description="SAM valence before session")
    pre_arousal: int = Field(..., ge=1, le=9, description="SAM arousal before session")
    post_valence: int = Field(..., ge=1, le=9, description="SAM valence after session")
    post_arousal: int = Field(..., ge=1, le=9, description="SAM arousal after session")
    completed: bool = True
    skipped_tracks: Optional[List[str]] = Field(default_factory=list)
    user_id: str = Field("default_user", description="Identifier for personalization")


class RerouteRequest(BaseModel):
    session_id: str
    position: int = Field(..., ge=1, description="1-indexed position of track to replace")
    user_id: str = Field("default_user", description="Identifier for personalization")


class WaypointInput(BaseModel):
    valence: float = Field(..., ge=0.0, le=1.0)
    arousal: float = Field(..., ge=0.0, le=1.0)


class UpdateWaypointsRequest(BaseModel):
    session_id: str
    waypoints: List[WaypointInput] = Field(..., min_length=2, description="New ordered waypoints")
    user_id: str = Field("default_user")


# ── Responses ───────────────────────────────────────────────────────────────

class MoodInfo(BaseModel):
    label: str
    emoji: str
    valence: float
    arousal: float
    quadrant: int
    color: str


class WaypointInfo(BaseModel):
    index: int
    valence: float
    arousal: float


class TrackInfo(BaseModel):
    position: int
    track_id: str
    track_name: str
    artists: str
    album_name: str
    valence: float
    arousal: float
    genre: str
    waypoint_index: int
    spotify_url: str


class PlaylistMetrics(BaseModel):
    endpoint_accuracy: float
    smoothness_score: float
    max_jump: float
    iso_compliance: bool
    genre_entropy: float


class PlaylistResponse(BaseModel):
    session_id: str
    mood_start: MoodInfo
    mood_target: MoodInfo
    waypoints: List[WaypointInfo]
    playlist: List[TrackInfo]
    metrics: PlaylistMetrics


class FeedbackResponse(BaseModel):
    status: str = "ok"
    session_id: str
    direction_correct: Optional[bool] = None
    valence_delta: Optional[int] = None
    arousal_delta: Optional[int] = None


# ── Auth models ────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: str
    email: str
    provider: str
    name: str
    avatar_url: str
    created_at: str


class AuthStatusResponse(BaseModel):
    authenticated: bool
    user: Optional[UserResponse] = None


class GuestLoginRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional guest identifier")


# ── Saved Journeys ───────────────────────────────────────────────────────────

class SaveJourneyRequest(BaseModel):
    session_id: str
    name: str = "Untitled Journey"


class JourneySummary(BaseModel):
    id: str
    name: str
    session_id: str
    start_valence: float
    start_arousal: float
    target_valence: float
    target_arousal: float
    path_type: str
    track_count: int
    created_at: str


class JourneyDetail(BaseModel):
    id: str
    name: str
    session_id: str
    start_valence: float
    start_arousal: float
    target_valence: float
    target_arousal: float
    path_type: str
    mood_start_json: str
    mood_target_json: str
    waypoints_json: str
    playlist_json: str
    metrics_json: Optional[str]
    track_count: int
    created_at: str
