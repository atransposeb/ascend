"""
main.py – FastAPI application for Ascend.

Start with:
    cd emotional_journey
    uvicorn src.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional

print("GOOGLE_CLIENT_ID =", os.getenv("GOOGLE_CLIENT_ID"))
print("SPOTIFY_CLIENT_ID =", os.getenv("SPOTIFY_CLIENT_ID"))
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from .mood_grid import mood_grid_json, get_mood
from .models import (
    FeedbackRequest,
    FeedbackResponse,
    PlaylistRequest,
    PlaylistResponse,
    RerouteRequest,
    UpdateWaypointsRequest,
    UserResponse,
    AuthStatusResponse,
    GuestLoginRequest,
    SaveJourneyRequest,
    JourneySummary,
    JourneyDetail,
)
from .playlist_engine import generate_playlist, load_artifacts, reroute_track, update_segment_waypoints
from .database import (
    init_db, save_session, get_session, save_feedback,
    get_user_preferences, get_user_sessions,
    save_journey as db_save_journey,
    list_journeys as db_list_journeys,
    get_journey as db_get_journey,
    delete_journey as db_delete_journey,
)
from .auth import (
    oauth,
    get_current_user,
    set_auth_cookie,
    clear_auth_cookie,
    handle_oauth_callback,
    create_jwt,
    COOKIE_NAME,
    BASE_URL,
)

# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ascend",
    description="Intelligent emotional navigation through continuous Valence-Arousal space.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware, secret_key=os.getenv("ASCEND_JWT_SECRET", "ascend-dev-secret-change-in-production"))

# Serve the frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.on_event("startup")
def startup():
    init_db()
    try:
        load_artifacts()
    except FileNotFoundError:
        print("Data artifacts not found, running data pipeline...")
        from .data_pipeline import run_pipeline
        run_pipeline()
        load_artifacts()
    print("🚀 Ascend API ready.")


# ── Serve frontend ──────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the frontend index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


# ── Helper: resolve user_id from auth or request body ──────────────────────


def resolve_user_id(request: Request, body_user_id: Optional[str] = None) -> str:
    """Return the authenticated user's ID, or fall back to the request body value, or 'guest'."""
    user = get_current_user(request)
    if user:
        return user["id"]
    return body_user_id or "guest"


# ── Auth routes ─────────────────────────────────────────────────────────────


@app.get("/auth/login/{provider}")
async def login_oauth(provider: str, request: Request):
    """Initiate OAuth login with the given provider (google, spotify, github)."""
    if provider not in ("google", "spotify", "github"):
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    client = oauth.create_client(provider)
    if not client:
        raise HTTPException(status_code=501, detail=f"Provider {provider} is not configured")
    # Each provider has a different callback URI registered in their dashboard
    callback_map = {
        "google": f"{BASE_URL}/auth/callback",
        "spotify": f"{BASE_URL}/auth/spotify/callback",
        "github": f"{BASE_URL}/auth/callback/github",
    }
    redirect_uri = callback_map[provider]
    return await client.authorize_redirect(request, redirect_uri)


async def _finish_callback(provider: str, request: Request) -> RedirectResponse:
    """Complete OAuth flow and redirect to frontend."""
    try:
        user, token = await handle_oauth_callback(provider, request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth callback failed: {e}")
    response = RedirectResponse(url="/?login=success")
    set_auth_cookie(response, token)
    return response


@app.get("/auth/callback")  # Google — matches dashboard URI
async def callback_google(request: Request):
    return await _finish_callback("google", request)


@app.get("/auth/spotify/callback")  # Spotify — matches dashboard URI
async def callback_spotify(request: Request):
    return await _finish_callback("spotify", request)


@app.get("/auth/callback/github")  # GitHub (if configured)
async def callback_github(request: Request):
    return await _finish_callback("github", request)


@app.post("/auth/guest")
async def guest_login(response: Response, req: GuestLoginRequest = GuestLoginRequest()):
    """Create a guest session (no OAuth). Returns a JWT for the guest."""
    guest_id = req.user_id or f"guest_{uuid.uuid4().hex[:8]}"
    token = create_jwt(guest_id)
    set_auth_cookie(response, token)
    return AuthStatusResponse(
        authenticated=False,
        user=UserResponse(
            id=guest_id,
            email="",
            provider="guest",
            name="Guest",
            avatar_url="",
            created_at="",
        ),
    )


@app.post("/auth/logout")
async def logout(response: Response):
    """Clear the auth cookie."""
    clear_auth_cookie(response)
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthStatusResponse)
async def auth_me(request: Request):
    """Return the current authenticated user (if any)."""
    user = get_current_user(request)
    if user:
        return AuthStatusResponse(
            authenticated=True,
            user=UserResponse(
                id=user["id"],
                email=user["email"],
                provider=user["provider"],
                name=user["name"],
                avatar_url=user["avatar_url"],
                created_at=str(user["created_at"]) if user.get("created_at") else "",
            ),
        )
    return AuthStatusResponse(authenticated=False)


# ── OAuth helper to inject oauth into request state ─────────────────────────


@app.middleware("http")
async def attach_oauth_client(request: Request, call_next):
    """Make the OAuth client available during request handling."""
    request.state.oauth = oauth
    response = await call_next(request)
    return response


# ── API endpoints ───────────────────────────────────────────────────────────


@app.get("/api/moods")
def get_moods():
    """Return the 16-mood grid with coordinates, labels, and colours."""
    return mood_grid_json()


@app.post("/api/playlist/generate", response_model=PlaylistResponse)
def create_playlist(req: PlaylistRequest, request: Request):
    """Generate an emotional transition playlist."""
    user_id = resolve_user_id(request, req.user_id)
    try:
        response = generate_playlist(
            start_valence=req.start_valence,
            start_arousal=req.start_arousal,
            target_valence=req.target_valence,
            target_arousal=req.target_arousal,
            n_songs=req.n_songs,
            genre_filter=req.genre_filter,
            user_id=user_id,
            path_type=req.path_type,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    save_session(
        session_id=response.session_id,
        start_valence=req.start_valence,
        start_arousal=req.start_arousal,
        target_valence=req.target_valence,
        target_arousal=req.target_arousal,
        user_id=user_id,
        path_type=req.path_type,
        playlist_json=json.dumps([t.model_dump() for t in response.playlist]),
        metrics_json=response.metrics.model_dump_json(),
        original_waypoints_json=json.dumps([w.model_dump() for w in response.waypoints]),
    )

    return response


@app.post("/api/playlist/update-waypoints")
def update_waypoints(req: UpdateWaypointsRequest, request: Request):
    """Regenerate playlist segments given a new set of waypoints from the UI."""
    user_id = resolve_user_id(request, req.user_id)
    try:
        new_wps = [(w.valence, w.arousal) for w in req.waypoints]
        result = update_segment_waypoints(
            session_id=req.session_id,
            new_waypoints=new_wps,
            user_id=user_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/playlist/reroute")
def reroute_playlist_track(req: RerouteRequest, request: Request):
    """Replace a skipped track in a session with a suitable alternative."""
    user_id = resolve_user_id(request, req.user_id)
    try:
        result = reroute_track(
            session_id=req.session_id,
            position=req.position,
            user_id=user_id,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/playlist/{session_id}")
def retrieve_playlist(session_id: str):
    """Retrieve a previously generated playlist."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session["playlist"] = json.loads(session["playlist"])
    session["metrics"] = json.loads(session["metrics"]) if session.get("metrics") else None
    return session


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest, request: Request):
    """Store post-session SAM scale feedback."""
    user_id = resolve_user_id(request, req.user_id)
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    save_feedback(
        session_id=req.session_id,
        pre_valence=req.pre_valence,
        pre_arousal=req.pre_arousal,
        post_valence=req.post_valence,
        post_arousal=req.post_arousal,
        completed=req.completed,
        skipped_tracks=req.skipped_tracks or [],
        user_id=user_id,
    )

    v_delta = req.post_valence - req.pre_valence
    a_delta = req.post_arousal - req.pre_arousal

    target_v_dir = session["target_valence"] - session["start_valence"]
    target_a_dir = session["target_arousal"] - session["start_arousal"]

    v_correct = (v_delta * target_v_dir) >= 0 or abs(target_v_dir) < 0.05
    a_correct = (a_delta * target_a_dir) >= 0 or abs(target_a_dir) < 0.05
    direction_correct = v_correct and a_correct

    return FeedbackResponse(
        session_id=req.session_id,
        direction_correct=direction_correct,
        valence_delta=v_delta,
        arousal_delta=a_delta,
    )


@app.get("/api/user/preferences/{user_id}")
def get_user_prefs(user_id: str, request: Request):
    """Retrieve current personalization preferences for a user."""
    auth_user = get_current_user(request)
    effective_id = auth_user["id"] if auth_user else user_id
    return get_user_preferences(effective_id)


@app.get("/api/user/sessions")
def user_sessions(request: Request):
    """Return the current user's previous session history."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return get_user_sessions(user["id"])


# ── Saved Journeys endpoints ─────────────────────────────────────────────────


@app.post("/api/journeys/save", response_model=JourneyDetail)
def save_journey(req: SaveJourneyRequest, request: Request):
    """Save the current journey state to the user's history."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    user_id = user["id"]
    journey_id = uuid.uuid4().hex

    mood_start = {"valence": session["start_valence"], "arousal": session["start_arousal"]}
    mood_target = {"valence": session["target_valence"], "arousal": session["target_arousal"]}

    waypoints_str = session.get("original_waypoints")
    if not waypoints_str:
        waypoints_str = json.dumps([
            {"valence": session["start_valence"], "arousal": session["start_arousal"]},
            {"valence": session["target_valence"], "arousal": session["target_arousal"]},
        ])

    playlist = json.loads(session["playlist"]) if isinstance(session["playlist"], str) else session["playlist"]

    result = db_save_journey(
        journey_id=journey_id,
        user_id=user_id,
        name=req.name,
        session_id=req.session_id,
        start_valence=session["start_valence"],
        start_arousal=session["start_arousal"],
        target_valence=session["target_valence"],
        target_arousal=session["target_arousal"],
        path_type=session["path_type"],
        mood_start_json=json.dumps(mood_start),
        mood_target_json=json.dumps(mood_target),
        waypoints_json=waypoints_str,
        playlist_json=json.dumps(playlist),
        metrics_json=session.get("metrics"),
        track_count=len(playlist),
    )
    return result


@app.get("/api/journeys", response_model=list[JourneySummary])
def list_journeys(request: Request):
    """List all saved journeys for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return db_list_journeys(user["id"])


@app.get("/api/journeys/{journey_id}", response_model=JourneyDetail)
def get_journey(journey_id: str, request: Request):
    """Get full details of a saved journey."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    journey = db_get_journey(journey_id)
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey not found")
    if journey["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your journey")
    return journey


@app.delete("/api/journeys/{journey_id}")
def delete_journey(journey_id: str, request: Request):
    """Delete a saved journey."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    journey = db_get_journey(journey_id)
    if journey is None:
        raise HTTPException(status_code=404, detail="Journey not found")
    if journey["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Not your journey")
    db_delete_journey(journey_id)
    return {"status": "deleted"}
