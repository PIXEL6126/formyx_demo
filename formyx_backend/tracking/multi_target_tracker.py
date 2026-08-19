"""
formyx_backend/tracking/multi_target_tracker.py
-----------------------------------------------
Manages **multiple** simultaneous balloon/drone tracks using independent
3D Kalman Filters with nearest-neighbour association and track lifecycle
management (birth → active → coasting → dead).

Key design choices
------------------
* Each confirmed track gets its own TargetTracker (Kalman filter).
* Per-frame, detections are matched to existing tracks via **minimum
  Euclidean distance** in 3D camera-frame coordinates.
* A Hungarian-style greedy assignment keeps it simple and O(N·M).
* Unmatched detections start new tentative tracks.
* Tracks with no measurements for *coast_frames* ticks are retired.
* The primary balloon target (closest or highest-confidence) is always
  queryable via `get_primary_target()`.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import get
from tracking.target_tracker import TargetTracker

log = logging.getLogger(__name__)


class Track:
    """Wraps a TargetTracker with lifecycle metadata."""

    _id_counter = itertools.count(1)

    def __init__(self, initial_pos: Tuple[float, float, float], class_id: int, confidence: float) -> None:
        self.track_id: int = next(Track._id_counter)
        self.class_id: int = class_id
        self.confidence: float = confidence
        self.kalman = TargetTracker()
        self.kalman.update(initial_pos)
        self.hit_count: int = 1
        self.coast_count: int = 0
        self.is_confirmed: bool = False
        # A track is confirmed after N consecutive hits
        self._confirm_threshold: int = 2

    @property
    def position_3d(self) -> Optional[Tuple[float, float, float]]:
        state = self.kalman.get_state()
        if state is None:
            return None
        return (state[0], state[1], state[2])

    @property
    def state(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        return self.kalman.get_state()

    def predict(self, dt: float) -> None:
        self.kalman.predict(dt=dt)

    def update(self, measurement: Tuple[float, float, float], confidence: float) -> None:
        accepted = self.kalman.update(measurement)
        if accepted:
            self.hit_count += 1
            self.coast_count = 0
            self.confidence = max(self.confidence, confidence)
            if self.hit_count >= self._confirm_threshold:
                self.is_confirmed = True
        else:
            self.coast_count += 1

    def coast(self) -> None:
        """Called when no detection matches this track in the current frame."""
        self.coast_count += 1


class MultiTargetTracker:
    """
    Manages the full set of active target tracks.

    Parameters (loaded from config → tracking.*)
    -----------------------------------------------
    max_coast_frames      — delete a track after this many frames without update
    association_gate_m    — 3D Euclidean distance gate for matching (metres)
    """

    def __init__(self) -> None:
        self.max_coast: int = get("tracking", "max_lost_frames", 30)
        self.gate_m: float = get("tracking", "association_gate_m", 3.0)
        self.tracks: List[Track] = []
        log.info(
            "MultiTargetTracker initialised — max_coast=%d  gate=%.1fm",
            self.max_coast, self.gate_m,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate all tracks forward by *dt* seconds."""
        for track in self.tracks:
            track.predict(dt)

    def update(
        self,
        detections: List[Dict[str, Any]],
        dt: float = 1.0 / 30.0,
    ) -> None:
        """
        Full predict → associate → update → birth → prune cycle.

        Parameters
        ----------
        detections : list of dicts
            Each dict must have keys: rx, ry, rz (3D camera-frame metres),
            class_id (int), confidence (float).
            Detections with rz=None are skipped.
        dt : float
            Time step in seconds since the last call.
        """
        # 1. Predict existing tracks
        self.predict(dt)

        # 2. Filter detections that have valid 3D coordinates
        valid_dets = [d for d in detections if d.get("rz") is not None]

        # 3. Associate detections → tracks (greedy nearest-neighbour)
        matched_track_idxs: set = set()
        matched_det_idxs: set = set()

        if self.tracks and valid_dets:
            # Build cost matrix  (num_tracks × num_dets)
            cost = np.zeros((len(self.tracks), len(valid_dets)), dtype=np.float64)
            for ti, trk in enumerate(self.tracks):
                pos = trk.position_3d
                if pos is None:
                    cost[ti, :] = 1e9
                    continue
                for di, det in enumerate(valid_dets):
                    dx = pos[0] - det["rx"]
                    dy = pos[1] - det["ry"]
                    dz = pos[2] - det["rz"]
                    cost[ti, di] = np.sqrt(dx * dx + dy * dy + dz * dz)

            # Greedy assignment: pick smallest cost pairs
            flat_order = np.argsort(cost, axis=None)
            for flat_idx in flat_order:
                ti, di = divmod(int(flat_idx), len(valid_dets))
                if ti in matched_track_idxs or di in matched_det_idxs:
                    continue
                if cost[ti, di] > self.gate_m:
                    break  # remaining costs are all above gate
                # Match!
                det = valid_dets[di]
                self.tracks[ti].update(
                    (det["rx"], det["ry"], det["rz"]),
                    det["confidence"],
                )
                matched_track_idxs.add(ti)
                matched_det_idxs.add(di)

        # 4. Coast unmatched tracks
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_track_idxs:
                trk.coast()

        # 5. Birth new tracks for unmatched detections
        for di, det in enumerate(valid_dets):
            if di not in matched_det_idxs:
                new_track = Track(
                    initial_pos=(det["rx"], det["ry"], det["rz"]),
                    class_id=det.get("class_id", 0),
                    confidence=det["confidence"],
                )
                self.tracks.append(new_track)
                log.debug(
                    "New track #%d spawned at (%.2f, %.2f, %.2f)",
                    new_track.track_id, det["rx"], det["ry"], det["rz"],
                )

        # 6. Prune dead tracks
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.coast_count < self.max_coast]
        pruned = before - len(self.tracks)
        if pruned > 0:
            log.debug("Pruned %d dead tracks. Active: %d", pruned, len(self.tracks))

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Return only tracks that have been confirmed (≥ N consecutive hits)."""
        return [t for t in self.tracks if t.is_confirmed]

    @property
    def balloon_tracks(self) -> List[Track]:
        """Return confirmed tracks with class_id == 0 (balloon)."""
        return [t for t in self.confirmed_tracks if t.class_id == 0]

    @property
    def drone_tracks(self) -> List[Track]:
        """Return confirmed tracks with class_id == 1 (drone)."""
        return [t for t in self.confirmed_tracks if t.class_id == 1]

    @property
    def is_tracking(self) -> bool:
        """True if at least one confirmed balloon track exists."""
        return len(self.balloon_tracks) > 0

    def get_primary_target(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Return the state (x, y, z, vx, vy, vz) of the *primary* balloon target.

        Primary = closest confirmed balloon track (smallest z).
        """
        balloons = self.balloon_tracks
        if not balloons:
            return None
        # Pick the one with the smallest depth (closest)
        best = min(balloons, key=lambda t: (t.position_3d[2] if t.position_3d else 1e9))
        return best.state

    def get_all_states(self) -> List[Dict[str, Any]]:
        """
        Return a list of dicts for every confirmed track:
            track_id, class_id, confidence, state (x,y,z,vx,vy,vz)
        """
        results = []
        for t in self.confirmed_tracks:
            state = t.state
            if state is None:
                continue
            results.append({
                "track_id": t.track_id,
                "class_id": t.class_id,
                "confidence": t.confidence,
                "state": state,
            })
        return results

    @property
    def is_initialized(self) -> bool:
        """Backward-compatible flag for camcv.py / main.py HUD."""
        return self.is_tracking

    def get_state(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Backward-compatible accessor — returns primary balloon state."""
        return self.get_primary_target()

    def reset(self) -> None:
        """Drop all tracks."""
        self.tracks.clear()
        log.info("MultiTargetTracker reset — all tracks dropped.")
