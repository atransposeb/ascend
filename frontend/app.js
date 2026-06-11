/**
 * app.js – Ascend frontend application.
 *
 * State machine: LANDING → SELECT → DASHBOARD
 * Auth: OAuth (Google, Spotify, GitHub) + Guest via JWT cookies.
 */

(() => {
    'use strict';

    // ── API ────────────────────────────────────────────────────────────────
    const API = '';

    async function api(path, opts = {}) {
        const res = await fetch(`${API}${path}`, {
            credentials: 'include',
            headers: { 'Content-Type': 'application/json' },
            ...opts,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `API error ${res.status}`);
        }
        return res.json();
    }

    // ── Auth state ─────────────────────────────────────────────────────────
    let currentUser   = null;   // {id, email, provider, name, avatar_url} | null
    let isGuest       = false;

    // ── Application state ──────────────────────────────────────────────────
    let selectedPathType  = 'linear';
    let startCoords       = null;
    let targetCoords      = null;
    let selectionPhase    = 'start';
    let currentPlaylist   = null;
    let samValues         = { preV: null, preA: null, postV: null, postA: null };
    let activeTrackId     = null;

    // Draggable waypoint state
    let waypointsMutable  = [];
    let draggingWpIdx     = null;
    let isDragging        = false;
    let dragMoved         = false;

    // ── DOM refs ───────────────────────────────────────────────────────────

    // Screens
    const $screenLanding   = document.getElementById('screen-landing');
    const $screenSelect    = document.getElementById('screen-select');
    const $screenDashboard = document.getElementById('screen-dashboard');

    // Landing – auth
    const authBtns         = document.querySelectorAll('.auth-btn');
    const $btnGuest        = document.getElementById('btn-guest');

    // Landing – path cards
    const modeCards        = document.querySelectorAll('.mode-card');

    // Selection
    const $btnBackLanding  = document.getElementById('btn-back-landing');
    const $btnGenerate     = document.getElementById('btn-generate');
    const $selectorCanvas  = document.getElementById('mood-selector-canvas');
    const $instructionStep = document.getElementById('instruction-step');
    const $coordStartVal   = document.getElementById('coord-start-val');
    const $coordTargetVal  = document.getElementById('coord-target-val');
    const $pathModeDisplay = document.getElementById('path-mode-display');

    // Dashboard – left panel
    const $btnBackSelect   = document.getElementById('btn-back-select');
    const $vaCanvas        = document.getElementById('va-canvas');
    const $metricsGrid     = document.getElementById('metrics-grid');
    const $chartDragHint   = document.getElementById('chart-drag-hint');
    const tabBtns          = document.querySelectorAll('.panel-tab-btn');
    const tabContents      = document.querySelectorAll('.panel-tab');

    // Dashboard – user profile
    const $userProfile     = document.getElementById('user-profile');
    const $userAvatar      = document.getElementById('user-avatar');
    const $userName        = document.getElementById('user-name');
    const $btnSignout      = document.getElementById('btn-signout');

    // Dashboard – mobile view toggle
    const $dashboardToggle = document.getElementById('dashboard-toggle');
    const toggleBtns       = document.querySelectorAll('.toggle-btn');

    // Dashboard – right panel
    const $tracklist       = document.getElementById('tracklist');
    const $journeyLabel    = document.getElementById('journey-label');
    const $trackCount      = document.getElementById('track-count');
    const $btnSaveJourney  = document.getElementById('btn-save-journey');

    // Floating player panel
    const $playerFloating     = document.getElementById('player-floating');
    const $playerFloatingTrack= document.getElementById('player-floating-track');
    const $playerFloatingClose= document.getElementById('player-floating-close');
    const $playerFloatingIframe   = document.getElementById('player-floating-iframe');
    const $playerFloatingPlaceholder = document.getElementById('player-floating-placeholder');

    // History
    const $historyList   = document.getElementById('history-list');
    const $historyEmpty  = document.getElementById('history-empty');

    // Feedback
    const $btnSkipFeedback = document.getElementById('btn-skip-feedback');
    const $btnSubmitFeedback = document.getElementById('btn-submit-feedback');
    const $feedbackResult  = document.getElementById('feedback-result');
    const $resultDeltas    = document.getElementById('result-deltas');

    // Loader
    const $loader          = document.getElementById('loader');

    // ── Path type names ────────────────────────────────────────────────────
    const PATH_NAMES = {
        linear: 'Linear Chord',
        creative: 'Creative Arch',
        random: 'Exploration',
    };

    // ── Auth helpers ──────────────────────────────────────────────────────

    async function checkAuth() {
        try {
            const res = await api('/auth/me');
            if (res.authenticated && res.user) {
                currentUser = res.user;
                isGuest = res.user.provider === 'guest';
                renderUserProfile();
                return true;
            }
        } catch (_) {}
        currentUser = null;
        isGuest = false;
        return false;
    }

    function renderUserProfile() {
        if (currentUser) {
            if (currentUser.avatar_url) {
                $userAvatar.src = currentUser.avatar_url;
                $userAvatar.style.display = 'inline-block';
            }
            $userName.textContent = currentUser.name || currentUser.email || currentUser.provider;
            $btnSignout.style.display = 'inline-block';
        } else {
            $userAvatar.style.display = 'none';
            $userName.textContent = '';
            $btnSignout.style.display = 'none';
        }
    }

    async function handleGuestLogin() {
        try {
            await api('/auth/guest', { method: 'POST' });
            await checkAuth();
            proceedToSelection();
        } catch (err) {
            alert(`Guest login failed: ${err.message}`);
        }
    }

    async function handleSignOut() {
        try {
            await api('/auth/logout', { method: 'POST' });
        } catch (_) {}
        currentUser = null;
        isGuest = false;
        renderUserProfile();
        showScreen('screen-landing');
    }

    function handleOAuthClick(provider) {
        window.location.href = `/auth/login/${provider}`;
    }

    // ── Screen management ──────────────────────────────────────────────────

    function showScreen(id) {
        [$screenLanding, $screenSelect, $screenDashboard].forEach(el => {
            el.classList.toggle('active', el.id === id);
        });
        window.scrollTo({ top: 0, behavior: 'instant' });
    }

    function proceedToSelection() {
        $pathModeDisplay.textContent = PATH_NAMES[selectedPathType];
        showScreen('screen-select');
        setTimeout(drawMoodSelector, 100);
    }

    // ── URL param check for OAuth callback ────────────────────────────────

    async function handlePostLogin() {
        const params = new URLSearchParams(window.location.search);
        if (params.get('login') === 'success') {
            window.history.replaceState({}, document.title, window.location.pathname);
            const authed = await checkAuth();
            if (authed) {
                proceedToSelection();
                return;
            }
        }
        await checkAuth();
    }

    // ── Landing interactions ───────────────────────────────────────────────

    authBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            handleOAuthClick(btn.dataset.provider);
        });
    });

    $btnGuest.addEventListener('click', handleGuestLogin);

    modeCards.forEach(card => {
        card.addEventListener('click', () => {
            modeCards.forEach(c => {
                c.classList.remove('active');
                c.querySelector('.card-select-indicator').textContent = 'SELECT';
            });
            card.classList.add('active');
            card.querySelector('.card-select-indicator').textContent = 'SELECTED';
            selectedPathType = card.dataset.path;
        });
    });

    $btnSignout.addEventListener('click', handleSignOut);

    // ── Mood Selector Canvas ───────────────────────────────────────────────

    $selectorCanvas.addEventListener('mousedown', onCanvasClick);

    function onCanvasClick(e) {
        const rect = $selectorCanvas.getBoundingClientRect();
        const pad  = 30;
        const w    = rect.width;
        const h    = rect.height;
        const plotW = w - pad * 2;
        const plotH = h - pad * 2;

        let v = (e.clientX - rect.left - pad) / plotW;
        let a = 1.0 - (e.clientY - rect.top - pad) / plotH;
        v = Math.max(0, Math.min(1, v));
        a = Math.max(0, Math.min(1, a));

        if (selectionPhase === 'start') {
            startCoords = { v, a };
            targetCoords = null;
            selectionPhase = 'target';
            $instructionStep.innerHTML = `
                <span class="step-num">02</span>
                <span class="step-text">Now click to place your <strong>target</strong> emotional state</span>
            `;
            $coordStartVal.textContent = fmtCoord(v, a);
            $coordTargetVal.textContent = '—';
            $btnGenerate.disabled = true;
        } else {
            targetCoords = { v, a };
            selectionPhase = 'start';
            $instructionStep.innerHTML = `
                <span class="step-num">✓</span>
                <span class="step-text">Both states placed. Click <strong>Generate</strong> to begin, or re-click to adjust.</span>
            `;
            $coordTargetVal.textContent = fmtCoord(v, a);
            $btnGenerate.disabled = false;
        }
        drawMoodSelector();
    }

    function fmtCoord(v, a) {
        return `V ${v.toFixed(2)} · A ${a.toFixed(2)}`;
    }

    function drawMoodSelector() {
        if (!$selectorCanvas) return;
        const dpr  = window.devicePixelRatio || 1;
        const rect = $selectorCanvas.getBoundingClientRect();
        const w    = rect.width;
        const h    = rect.width;
        $selectorCanvas.width  = w * dpr;
        $selectorCanvas.height = h * dpr;
        $selectorCanvas.style.height = `${h}px`;
        const ctx = $selectorCanvas.getContext('2d');
        ctx.scale(dpr, dpr);

        drawVAPlane(ctx, w, h, {
            waypoints: null,
            songs: null,
            start: startCoords,
            target: targetCoords,
            pathType: selectedPathType,
            isSelector: true,
        });
    }

    // ── Shared VA plane renderer ───────────────────────────────────────────

    function drawVAPlane(ctx, w, h, opts = {}) {
        const pad   = 30;
        const plotW = w - pad * 2;
        const plotH = h - pad * 2;
        const toX   = v => pad + v * plotW;
        const toY   = a => pad + (1 - a) * plotH;

        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = 'rgba(0,0,0,0.06)';
        ctx.setLineDash([1, 3]);
        ctx.lineWidth = 0.5;
        for (let s = 0.1; s < 1; s += 0.1) {
            if (Math.abs(s - 0.5) < 0.01) continue;
            ctx.beginPath(); ctx.moveTo(toX(s), toY(1)); ctx.lineTo(toX(s), toY(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(toX(0), toY(s)); ctx.lineTo(toX(1), toY(s)); ctx.stroke();
        }
        ctx.setLineDash([]);

        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(toX(0.5), toY(1.02)); ctx.lineTo(toX(0.5), toY(-0.02)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(toX(-0.02), toY(0.5)); ctx.lineTo(toX(1.02), toY(0.5)); ctx.stroke();

        ctx.font = 'italic 700 8px serif';
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.textAlign = 'center';
        ctx.fillText('Happy · Excited', toX(0.75), toY(0.87));
        ctx.fillText('Tense · Angry',   toX(0.25), toY(0.87));
        ctx.fillText('Sad · Low',       toX(0.25), toY(0.13));
        ctx.fillText('Calm · Relaxed',  toX(0.75), toY(0.13));

        ctx.fillStyle = 'rgba(0,0,0,0.35)';
        ctx.font = '700 7px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('VALENCE →', w / 2, h - 6);
        ctx.save();
        ctx.translate(10, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('AROUSAL →', 0, 0);
        ctx.restore();

        if (opts.isSelector && opts.start && opts.target) {
            drawTrajectoryLine(ctx, toX, toY, opts.start, opts.target, opts.pathType);
        }

        if (!opts.isSelector && opts.waypoints && opts.waypoints.length > 1) {
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            opts.waypoints.forEach((wp, i) => {
                const x = toX(wp.valence), y = toY(wp.arousal);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }

        if (!opts.isSelector && opts.waypoints) {
            opts.waypoints.forEach((wp, i) => {
                const x = toX(wp.valence);
                const y = toY(wp.arousal);
                const isGrabbed = opts.draggingIdx === i;
                const r = isGrabbed ? 9 : 6;
                ctx.beginPath();
                ctx.arc(x, y, r, 0, Math.PI * 2);
                ctx.fillStyle = '#FFFFFF';
                ctx.fill();
                ctx.strokeStyle = isGrabbed ? '#000000' : 'rgba(0,0,0,0.55)';
                ctx.lineWidth   = isGrabbed ? 2 : 1.5;
                ctx.stroke();
                if (isGrabbed) {
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, Math.PI * 2);
                    ctx.fillStyle = '#000000';
                    ctx.fill();
                }
            });
        }

        if (!opts.isSelector && opts.songs && opts.songs.length > 1) {
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2;
            ctx.beginPath();
            opts.songs.forEach((s, i) => {
                const x = toX(s.valence), y = toY(s.arousal);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        if (!opts.isSelector && opts.songs) {
            opts.songs.forEach((s, i) => {
                const x = toX(s.valence);
                const y = toY(s.arousal);
                const isActive = s.track_id === activeTrackId;
                ctx.fillStyle   = isActive ? '#000000' : '#333333';
                ctx.strokeStyle = '#000000';
                ctx.lineWidth   = 1;
                ctx.beginPath();
                ctx.rect(x - 6, y - 6, 12, 12);
                ctx.fill();
                ctx.fillStyle = '#FFFFFF';
                ctx.font = `800 ${isActive ? '8' : '7'}px monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(i + 1, x, y);
                ctx.textBaseline = 'alphabetic';
            });
        }

        if (opts.start)  drawSquareMarker(ctx, toX(opts.start.v),  toY(opts.start.a),  'S');
        if (opts.target) drawSquareMarker(ctx, toX(opts.target.v), toY(opts.target.a), 'T', true);

        if (!opts.isSelector && opts.startInfo) {
            drawSquareMarker(ctx, toX(opts.startInfo.valence),  toY(opts.startInfo.arousal),  'S');
        }
        if (!opts.isSelector && opts.targetInfo) {
            drawSquareMarker(ctx, toX(opts.targetInfo.valence), toY(opts.targetInfo.arousal), 'T', true);
        }
    }

    function drawTrajectoryLine(ctx, toX, toY, start, target, pathType) {
        ctx.strokeStyle = 'rgba(0,0,0,0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        if (pathType === 'creative') {
            const dv = target.v - start.v;
            const da = target.a - start.a;
            const dist = Math.sqrt(dv * dv + da * da);
            if (dist >= 0.1) {
                const mv = start.v + dv / 2, ma = start.a + da / 2;
                const pv = -da / dist, pa = dv / dist;
                const cv = Math.max(0, Math.min(1, mv + 0.25 * pv));
                const ca = Math.max(0, Math.min(1, ma + 0.25 * pa));
                ctx.moveTo(toX(start.v), toY(start.a));
                ctx.quadraticCurveTo(toX(cv), toY(ca), toX(target.v), toY(target.a));
            } else {
                ctx.moveTo(toX(start.v), toY(start.a));
                ctx.lineTo(toX(target.v), toY(target.a));
            }
        } else {
            ctx.moveTo(toX(start.v), toY(start.a));
            ctx.lineTo(toX(target.v), toY(target.a));
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function drawSquareMarker(ctx, x, y, label, filled = false) {
        ctx.fillStyle = filled ? '#000000' : '#FFFFFF';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.rect(x - 10, y - 10, 20, 20);
        ctx.fill();
        ctx.stroke();
        ctx.font = '800 8px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = filled ? '#FFFFFF' : '#000000';
        ctx.fillText(label, x, y);
        ctx.textBaseline = 'alphabetic';
    }

    // ── Playlist generation ────────────────────────────────────────────────

    async function generatePlaylist() {
        if (!startCoords || !targetCoords) return;
        $loader.classList.add('visible');
        $btnGenerate.disabled = true;
        try {
            currentPlaylist = await api('/api/playlist/generate', {
                method: 'POST',
                body: JSON.stringify({
                    start_valence:  startCoords.v,
                    start_arousal:  startCoords.a,
                    target_valence: targetCoords.v,
                    target_arousal: targetCoords.a,
                    n_songs:  15,
                    path_type: selectedPathType,
                    user_id:  'frontend',
                }),
            });
            activeTrackId = null;
            renderDashboard();
            showScreen('screen-dashboard');
        } catch (err) {
            alert(`Failed to generate playlist: ${err.message}`);
            $btnGenerate.disabled = false;
        } finally {
            $loader.classList.remove('visible');
        }
    }

    // ── Draggable Waypoint System ──────────────────────────────────────────

    function vaCanvasLayout() {
        const rect  = $vaCanvas.getBoundingClientRect();
        const pad   = 30;
        const w     = rect.width;
        const h     = rect.height;
        return { rect, pad, w, plotW: w - pad * 2, plotH: h - pad * 2 };
    }

    function vaToScreenXY(v, a) {
        const { pad, plotW, plotH } = vaCanvasLayout();
        return { x: pad + v * plotW, y: pad + (1 - a) * plotH };
    }

    function clientToVA(clientX, clientY) {
        const { rect, pad, plotW, plotH } = vaCanvasLayout();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        return {
            v: Math.max(0, Math.min(1, (x - pad) / plotW)),
            a: Math.max(0, Math.min(1, 1 - (y - pad) / plotH)),
        };
    }

    function hitTestWaypoints(clientX, clientY) {
        const { rect } = vaCanvasLayout();
        const mx = clientX - rect.left;
        const my = clientY - rect.top;
        const HIT = 14;
        for (let i = 0; i < waypointsMutable.length; i++) {
            const { x, y } = vaToScreenXY(waypointsMutable[i].valence, waypointsMutable[i].arousal);
            if (Math.hypot(mx - x, my - y) < HIT) return i;
        }
        return -1;
    }

    function onVAMouseHover(e) {
        if (isDragging) return;
        const idx = hitTestWaypoints(e.clientX, e.clientY);
        $vaCanvas.style.cursor = idx >= 0 ? 'grab' : 'default';
    }

    function onVAMouseDown(e) {
        const idx = hitTestWaypoints(e.clientX, e.clientY);
        if (idx < 0) return;
        draggingWpIdx = idx;
        isDragging    = true;
        dragMoved     = false;
        $vaCanvas.style.cursor = 'grabbing';
        if ($chartDragHint) $chartDragHint.classList.add('active');
        e.preventDefault();
    }

    function onVAMouseDrag(e) {
        if (!isDragging || draggingWpIdx === null) return;
        dragMoved = true;
        const { v, a } = clientToVA(e.clientX, e.clientY);
        waypointsMutable = waypointsMutable.map((wp, i) =>
            i === draggingWpIdx ? { ...wp, valence: v, arousal: a } : wp
        );
        renderVAChartLive(waypointsMutable);
    }

    async function onVAMouseUp() {
        if (!isDragging) return;
        isDragging    = false;
        draggingWpIdx = null;
        $vaCanvas.style.cursor = 'default';
        if ($chartDragHint) $chartDragHint.classList.remove('active');

        if (dragMoved && currentPlaylist) {
            dragMoved = false;
            try {
                $loader.classList.add('visible');
                const res = await api('/api/playlist/update-waypoints', {
                    method: 'POST',
                    body: JSON.stringify({
                        session_id: currentPlaylist.session_id,
                        waypoints:  waypointsMutable.map(wp => ({
                            valence: parseFloat(wp.valence.toFixed(4)),
                            arousal: parseFloat(wp.arousal.toFixed(4)),
                        })),
                        user_id: 'frontend',
                    }),
                });
                currentPlaylist.playlist  = res.playlist;
                currentPlaylist.waypoints = res.waypoints;
                currentPlaylist.metrics   = res.metrics;
                waypointsMutable = res.waypoints.map(wp => ({ ...wp }));
                renderTracklist();
                renderVAChartLive(waypointsMutable);
                renderMetrics();
            } catch (err) {
                alert(`Could not update trajectory: ${err.message}`);
                waypointsMutable = (currentPlaylist.waypoints || []).map(wp => ({ ...wp }));
                renderVAChartLive(waypointsMutable);
            } finally {
                $loader.classList.remove('visible');
            }
        }
        dragMoved = false;
    }

    function onVATouchStart(e) {
        if (e.touches.length !== 1) return;
        const t   = e.touches[0];
        const idx = hitTestWaypoints(t.clientX, t.clientY);
        if (idx < 0) return;
        draggingWpIdx = idx;
        isDragging    = true;
        dragMoved     = false;
        if ($chartDragHint) $chartDragHint.classList.add('active');
        e.preventDefault();
    }

    function onVATouchMove(e) {
        if (!isDragging || e.touches.length !== 1) return;
        e.preventDefault();
        const t = e.touches[0];
        dragMoved = true;
        const { v, a } = clientToVA(t.clientX, t.clientY);
        waypointsMutable = waypointsMutable.map((wp, i) =>
            i === draggingWpIdx ? { ...wp, valence: v, arousal: a } : wp
        );
        renderVAChartLive(waypointsMutable);
    }

    function initWaypointDrag() {
        $vaCanvas.addEventListener('mousedown',  onVAMouseDown);
        $vaCanvas.addEventListener('mousemove',  onVAMouseHover);
        document.addEventListener('mousemove',   onVAMouseDrag);
        document.addEventListener('mouseup',     onVAMouseUp);
        $vaCanvas.addEventListener('touchstart', onVATouchStart, { passive: false });
        $vaCanvas.addEventListener('touchmove',  onVATouchMove,  { passive: false });
        $vaCanvas.addEventListener('touchend',   onVAMouseUp);
    }

    function renderDashboard() {
        renderTracklist();
        renderMetrics();
        initSAMScales();
        closePlayerPanel();
        loadHistory();
        $btnSaveJourney.classList.remove('saved');
        $btnSaveJourney.textContent = 'Save';
        $feedbackResult.classList.remove('visible');
        samValues = { preV: null, preA: null, postV: null, postA: null };
        switchTab('chart');
        switchMobileView('chart');
        renderVAChart();
    }

    function renderTracklist() {
        if (!currentPlaylist) return;
        const { playlist, waypoints } = currentPlaylist;

        $journeyLabel.textContent = `${playlist.length} tracks`;
        $trackCount.textContent   = `${PATH_NAMES[selectedPathType] || 'Journey'} · DRAG HANDLES TO RESHAPE`;

        $tracklist.innerHTML = '';
        let lastWp = -1;

        playlist.forEach(track => {
            if (track.waypoint_index !== lastWp) {
                lastWp = track.waypoint_index;
                const progress = waypoints.length > 1
                    ? Math.round((track.waypoint_index / (waypoints.length - 1)) * 100)
                    : 100;
                const div = document.createElement('div');
                div.className = 'waypoint-divider';
                div.innerHTML = `<span class="wp-label">STAGE ${track.waypoint_index + 1} &mdash; ${progress}% OF JOURNEY</span>`;
                $tracklist.appendChild(div);
            }

            const card = buildTrackCard(track);
            $tracklist.appendChild(card);
        });
    }

    function buildTrackCard(track) {
        const card = document.createElement('div');
        card.className = 'track-card';
        if (track.track_id === activeTrackId) card.classList.add('now-playing');

        card.innerHTML = `
            <span class="track-number">${track.position}</span>
            <div class="track-info">
                <div class="track-name">${escHtml(track.track_name)}</div>
                <div class="track-artist">${escHtml(track.artists)}</div>
                <div class="track-genre">${escHtml(track.genre)}</div>
            </div>
            <div class="track-va">
                <div class="va-bar" title="Valence ${(track.valence * 100).toFixed(0)}%">
                    <div class="va-bar-fill" style="width:${(track.valence * 100).toFixed(1)}%"></div>
                </div>
                <div class="va-bar-label">V</div>
                <div class="va-bar" title="Arousal ${(track.arousal * 100).toFixed(0)}%">
                    <div class="va-bar-fill" style="width:${(track.arousal * 100).toFixed(1)}%"></div>
                </div>
                <div class="va-bar-label">A</div>
            </div>
            <div class="track-actions">
                <button class="btn-skip-track" data-position="${track.position}">SKIP</button>
                <button class="btn-play-track" data-track-id="${track.track_id}" data-track-name="${escAttr(track.track_name)}">▶ PLAY</button>
            </div>
        `;

        card.querySelector('.track-info').addEventListener('click', () => {
            window.open(track.spotify_url, '_blank', 'noopener');
        });

        card.querySelector('.btn-skip-track').addEventListener('click', e => {
            e.stopPropagation();
            skipTrack(track.position, card);
        });

        card.querySelector('.btn-play-track').addEventListener('click', e => {
            e.stopPropagation();
            embedSpotifyTrack(track.track_id, track.track_name);
        });

        return card;
    }

    // ── Floating Player Panel ──────────────────────────────────────────────

    function embedSpotifyTrack(trackId, trackName) {
        activeTrackId = trackId;
        $playerFloatingPlaceholder.style.display = 'none';
        $playerFloatingIframe.style.display = 'block';
        $playerFloatingIframe.src = `https://open.spotify.com/embed/track/${trackId}?utm_source=generator&theme=0`;
        $playerFloatingTrack.textContent = trackName;
        $playerFloating.classList.add('open');

        document.querySelectorAll('.track-card').forEach(c => {
            const btn = c.querySelector('.btn-play-track');
            if (btn && btn.dataset.trackId === trackId) {
                c.classList.add('now-playing');
            } else {
                c.classList.remove('now-playing');
            }
        });

        renderVAChart();
    }

    function closePlayerPanel() {
        $playerFloating.classList.remove('open');
        $playerFloatingIframe.src = '';
        $playerFloatingIframe.style.display = 'none';
        $playerFloatingPlaceholder.style.display = 'flex';
        $playerFloatingTrack.textContent = '—';
        activeTrackId = null;

        document.querySelectorAll('.track-card').forEach(c => {
            c.classList.remove('now-playing');
        });
        renderVAChart();
    }

    $playerFloatingClose.addEventListener('click', closePlayerPanel);

    // ── Skip / Reroute ─────────────────────────────────────────────────────

    async function skipTrack(position, cardEl) {
        if (!currentPlaylist) return;
        const skipBtn = cardEl.querySelector('.btn-skip-track');
        skipBtn.textContent = '…';
        skipBtn.disabled = true;
        cardEl.style.opacity = '0.45';

        try {
            const res = await api('/api/playlist/reroute', {
                method: 'POST',
                body: JSON.stringify({
                    session_id: currentPlaylist.session_id,
                    position,
                    user_id: 'frontend',
                }),
            });

            currentPlaylist.playlist[position - 1] = res.track;
            currentPlaylist.metrics = res.metrics;

            const newCard = buildTrackCard(res.track);
            cardEl.style.transition = 'opacity 0.2s';
            cardEl.style.opacity = '0';
            setTimeout(() => {
                cardEl.replaceWith(newCard);
                newCard.style.opacity = '0';
                newCard.style.transition = 'opacity 0.25s';
                requestAnimationFrame(() => { newCard.style.opacity = '1'; });
            }, 200);

            renderVAChart();
            renderMetrics();
        } catch (err) {
            alert(`Failed to replace track: ${err.message}`);
            skipBtn.textContent = 'SKIP';
            skipBtn.disabled = false;
            cardEl.style.opacity = '1';
        }
    }

    // ── VA Chart ───────────────────────────────────────────────────────────

    function renderVAChart() {
        if (!currentPlaylist) return;
        waypointsMutable = (currentPlaylist.waypoints || []).map(wp => ({ ...wp }));
        renderVAChartLive(waypointsMutable);
    }

    function renderVAChartLive(wps) {
        if (!$vaCanvas || !currentPlaylist) return;
        const dpr  = window.devicePixelRatio || 1;
        const rect = $vaCanvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;
        const w    = rect.width;
        const h    = rect.height;
        $vaCanvas.width  = w * dpr;
        $vaCanvas.height = h * dpr;
        const ctx  = $vaCanvas.getContext('2d');
        ctx.scale(dpr, dpr);

        drawVAPlane(ctx, w, h, {
            waypoints:   wps,
            songs:       currentPlaylist.playlist,
            startInfo:   currentPlaylist.mood_start,
            targetInfo:  currentPlaylist.mood_target,
            isSelector:  false,
            draggingIdx: draggingWpIdx,
        });
    }

    // ── Metrics ────────────────────────────────────────────────────────────

    function renderMetrics() {
        if (!currentPlaylist) return;
        const m = currentPlaylist.metrics;

        const items = [
            {
                name: 'ENDPOINT ACCURACY',
                val: m.endpoint_accuracy.toFixed(3),
                quality: m.endpoint_accuracy < 0.15 ? 'good' : m.endpoint_accuracy < 0.25 ? 'warn' : 'bad',
                badge: m.endpoint_accuracy < 0.15 ? 'GOOD' : m.endpoint_accuracy < 0.25 ? 'OK' : 'POOR',
            },
            {
                name: 'SMOOTHNESS',
                val: m.smoothness_score.toFixed(3),
                quality: m.smoothness_score < 0.15 ? 'good' : m.smoothness_score < 0.25 ? 'warn' : 'bad',
                badge: m.smoothness_score < 0.15 ? 'SMOOTH' : m.smoothness_score < 0.25 ? 'OK' : 'ROUGH',
            },
            {
                name: 'MAX JUMP',
                val: m.max_jump.toFixed(3),
                quality: m.max_jump < 0.3 ? 'good' : m.max_jump < 0.4 ? 'warn' : 'bad',
                badge: m.max_jump < 0.3 ? 'LOW' : m.max_jump < 0.4 ? 'MED' : 'HIGH',
            },
            {
                name: 'ISO COMPLIANCE',
                val: m.iso_compliance ? '✓' : '✗',
                quality: m.iso_compliance ? 'good' : 'bad',
                badge: m.iso_compliance ? 'PASS' : 'FAIL',
            },
            {
                name: 'GENRE ENTROPY',
                val: m.genre_entropy.toFixed(2),
                quality: m.genre_entropy > 1.5 ? 'good' : m.genre_entropy > 1.0 ? 'warn' : 'bad',
                badge: m.genre_entropy > 1.5 ? 'DIVERSE' : m.genre_entropy > 1.0 ? 'OK' : 'NARROW',
            },
        ];

        $metricsGrid.innerHTML = items.map(item => `
            <div class="metric-card">
                <div class="metric-name">${item.name}</div>
                <div class="metric-value ${item.quality}">${item.val}</div>
                <div class="metric-badge ${item.quality}">${item.badge}</div>
            </div>
        `).join('');
    }

    // ── Saved Journeys ──────────────────────────────────────────────────────

    async function saveJourney() {
        if (!currentPlaylist || !currentUser || isGuest) {
            if (!currentUser || isGuest) alert('Sign in to save journeys');
            return;
        }
        const name = prompt('Name this journey:', `Journey ${new Date().toLocaleDateString()}`);
        if (!name) return;

        $btnSaveJourney.textContent = 'Saving…';
        $btnSaveJourney.disabled = true;
        try {
            const res = await api('/api/journeys/save', {
                method: 'POST',
                body: JSON.stringify({
                    session_id: currentPlaylist.session_id,
                    name,
                }),
            });
            $btnSaveJourney.classList.add('saved');
            $btnSaveJourney.textContent = 'Saved ✓';
            renderHistory();
        } catch (err) {
            alert(`Save failed: ${err.message}`);
            $btnSaveJourney.classList.remove('saved');
            $btnSaveJourney.textContent = 'Save';
        } finally {
            $btnSaveJourney.disabled = false;
        }
    }

    async function loadJourney(journeyId) {
        try {
            $loader.classList.add('visible');
            const journey = await api(`/api/journeys/${journeyId}`);
            currentPlaylist = {
                session_id: journey.session_id,
                mood_start: JSON.parse(journey.mood_start_json),
                mood_target: JSON.parse(journey.mood_target_json),
                waypoints: JSON.parse(journey.waypoints_json),
                playlist: JSON.parse(journey.playlist_json),
                metrics: journey.metrics_json ? JSON.parse(journey.metrics_json) : null,
            };
            selectedPathType = journey.path_type;
            startCoords = { v: journey.start_valence, a: journey.start_arousal };
            targetCoords = { v: journey.target_valence, a: journey.target_arousal };
            activeTrackId = null;
            closePlayerPanel();
            renderDashboard();
            switchTab('chart');
        } catch (err) {
            alert(`Failed to load journey: ${err.message}`);
        } finally {
            $loader.classList.remove('visible');
        }
    }

    async function deleteJourney(journeyId, event) {
        event.stopPropagation();
        if (!confirm('Delete this saved journey?')) return;
        try {
            await api(`/api/journeys/${journeyId}`, { method: 'DELETE' });
            renderHistory();
        } catch (err) {
            alert(`Delete failed: ${err.message}`);
        }
    }

    async function loadHistory() {
        if (!currentUser || isGuest) {
            $historyEmpty.style.display = 'flex';
            $historyList.innerHTML = '';
            return;
        }
        try {
            const journeys = await api('/api/journeys');
            renderHistoryList(journeys);
        } catch (_) {
            $historyEmpty.style.display = 'flex';
        }
    }

    function renderHistory() {
        loadHistory();
    }

    function renderHistoryList(journeys) {
        $historyList.innerHTML = '';
        if (!journeys || journeys.length === 0) {
            $historyEmpty.style.display = 'flex';
            return;
        }
        $historyEmpty.style.display = 'none';
        journeys.forEach(j => {
            const item = document.createElement('div');
            item.className = 'history-item';
            item.addEventListener('click', () => loadJourney(j.id));

            const thumb = document.createElement('div');
            thumb.className = 'history-item-thumb';
            const canvas = document.createElement('canvas');
            canvas.width = 72;
            canvas.height = 72;
            thumb.appendChild(canvas);
            drawMiniVAChart(canvas, j);

            const body = document.createElement('div');
            body.className = 'history-item-body';
            body.innerHTML = `
                <div class="history-item-name">${escHtml(j.name)}</div>
                <div class="history-item-meta">${j.track_count} tracks · ${formatDate(j.created_at)} · ${j.path_type}</div>
            `;

            const delBtn = document.createElement('button');
            delBtn.className = 'history-item-delete';
            delBtn.textContent = '✕';
            delBtn.addEventListener('click', (e) => deleteJourney(j.id, e));

            item.appendChild(thumb);
            item.appendChild(body);
            item.appendChild(delBtn);
            $historyList.appendChild(item);
        });
    }

    function drawMiniVAChart(canvas, journey) {
        const ctx = canvas.getContext('2d');
        const w = 72, h = 72, pad = 6;
        const plotW = w - pad * 2, plotH = h - pad * 2;
        const toX = v => pad + v * plotW;
        const toY = a => pad + (1 - a) * plotH;

        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, w, h);

        ctx.strokeStyle = 'rgba(0,0,0,0.06)';
        ctx.lineWidth = 0.5;
        for (let s = 0.25; s < 1; s += 0.25) {
            ctx.beginPath(); ctx.moveTo(toX(s), toY(1)); ctx.lineTo(toX(s), toY(0)); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(toX(0), toY(s)); ctx.lineTo(toX(1), toY(s)); ctx.stroke();
        }

        let waypoints;
        try {
            waypoints = JSON.parse(journey.waypoints_json || '[]');
        } catch (_) {
            waypoints = [];
        }

        if (waypoints.length > 1) {
            ctx.strokeStyle = 'rgba(0,0,0,0.2)';
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            waypoints.forEach((wp, i) => {
                const x = toX(wp.valence || wp.v || 0);
                const y = toY(wp.arousal || wp.a || 0);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }

        ctx.fillStyle = '#000000';
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.rect(toX(journey.start_valence) - 5, toY(journey.start_arousal) - 5, 10, 10);
        ctx.fill();
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '700 5px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('S', toX(journey.start_valence), toY(journey.start_arousal));
        ctx.textBaseline = 'alphabetic';

        ctx.fillStyle = '#000000';
        ctx.beginPath();
        ctx.rect(toX(journey.target_valence) - 5, toY(journey.target_arousal) - 5, 10, 10);
        ctx.fill();
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '700 5px monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('T', toX(journey.target_valence), toY(journey.target_arousal));
        ctx.textBaseline = 'alphabetic';
    }

    function formatDate(dateStr) {
        if (!dateStr) return '';
        const d = new Date(dateStr);
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }

    $btnSaveJourney.addEventListener('click', saveJourney);

    // ── Tab management ─────────────────────────────────────────────────────

    function switchTab(tabName) {
        tabBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabName));
        tabContents.forEach(el => el.classList.toggle('active', el.id === `tab-content-${tabName}`));
        if (tabName === 'chart') {
            requestAnimationFrame(() => renderVAChart());
        }
        if (tabName === 'history') {
            renderHistory();
        }
    }

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // ── Mobile view toggle ──────────────────────────────────────────────────

    function switchMobileView(view) {
        if (!$dashboardToggle) return;
        toggleBtns.forEach(b => b.classList.toggle('active', b.dataset.view === view));
        $dashboardToggle.classList.remove('view-chart', 'view-tracks');
        $dashboardToggle.classList.add(`view-${view}`);
        if (view === 'chart') requestAnimationFrame(renderVAChart);
    }

    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => switchMobileView(btn.dataset.view));
    });

    // ── SAM Scales ─────────────────────────────────────────────────────────

    function initSAMScales() {
        const groups = [
            { id: 'sam-pre-valence',  key: 'preV'  },
            { id: 'sam-pre-arousal',  key: 'preA'  },
            { id: 'sam-post-valence', key: 'postV' },
            { id: 'sam-post-arousal', key: 'postA' },
        ];
        groups.forEach(({ id, key }) => {
            const container = document.getElementById(id);
            container.innerHTML = '';
            for (let i = 1; i <= 9; i++) {
                const btn = document.createElement('button');
                btn.className = 'sam-btn';
                btn.textContent = i;
                btn.addEventListener('click', () => {
                    container.querySelectorAll('.sam-btn').forEach(b => b.classList.remove('selected'));
                    btn.classList.add('selected');
                    samValues[key] = i;
                });
                container.appendChild(btn);
            }
        });
    }

    // ── Submit feedback ────────────────────────────────────────────────────

    async function submitFeedback() {
        const { preV, preA, postV, postA } = samValues;
        if (preV == null || preA == null || postV == null || postA == null) {
            alert('Please rate all four scales before submitting.');
            return;
        }
        try {
            const res = await api('/api/feedback', {
                method: 'POST',
                body: JSON.stringify({
                    session_id:   currentPlaylist.session_id,
                    pre_valence:  preV,
                    pre_arousal:  preA,
                    post_valence: postV,
                    post_arousal: postA,
                    completed:    true,
                    skipped_tracks: [],
                    user_id: 'frontend',
                }),
            });
            renderFeedbackResult(res);
        } catch (err) {
            alert(`Error submitting feedback: ${err.message}`);
        }
    }

    function renderFeedbackResult(res) {
        const vD = res.valence_delta ?? 0;
        const aD = res.arousal_delta ?? 0;
        $resultDeltas.innerHTML = `
            <div class="delta-item">
                <div class="delta-label">VALENCE Δ</div>
                <div class="delta-val">${vD > 0 ? '+' : ''}${vD}</div>
            </div>
            <div class="delta-item">
                <div class="delta-label">AROUSAL Δ</div>
                <div class="delta-val">${aD > 0 ? '+' : ''}${aD}</div>
            </div>
            <div class="delta-item">
                <div class="delta-label">DIRECTION</div>
                <div class="delta-val">${res.direction_correct ? '✓' : '✗'}</div>
            </div>
        `;
        $feedbackResult.classList.add('visible');
    }

    // ── Reset / navigation ─────────────────────────────────────────────────

    function resetSelection() {
        startCoords   = null;
        targetCoords  = null;
        selectionPhase = 'start';
        $instructionStep.innerHTML = `
            <span class="step-num">01</span>
            <span class="step-text">Click to place your <strong>current</strong> emotional state</span>
        `;
        $coordStartVal.textContent  = '—';
        $coordTargetVal.textContent = '—';
        $btnGenerate.disabled = true;
        drawMoodSelector();
    }

    $btnBackLanding.addEventListener('click', () => {
        resetSelection();
        showScreen('screen-landing');
    });

    $btnBackSelect.addEventListener('click', () => {
        resetSelection();
        currentPlaylist = null;
        activeTrackId   = null;
        samValues = { preV: null, preA: null, postV: null, postA: null };
        showScreen('screen-select');
        setTimeout(drawMoodSelector, 100);
    });

    $btnGenerate.addEventListener('click', generatePlaylist);
    $btnSubmitFeedback.addEventListener('click', submitFeedback);
    $btnSkipFeedback.addEventListener('click', () => {
        switchTab('chart');
    });

    // ── Resize handler ─────────────────────────────────────────────────────

    window.addEventListener('resize', () => {
        drawMoodSelector();
        if (currentPlaylist) renderVAChart();
    });

    // ── Helpers ────────────────────────────────────────────────────────────

    function escHtml(s) {
        const d = document.createElement('div');
        d.textContent = s ?? '';
        return d.innerHTML;
    }

    function escAttr(s) {
        return (s ?? '').replace(/"/g, '&quot;');
    }

    // ── Init ───────────────────────────────────────────────────────────────

    initSAMScales();
    initWaypointDrag();
    handlePostLogin();

})();
