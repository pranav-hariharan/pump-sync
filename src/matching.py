"""Track matching and playlist assembly"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, PHASE_CLUSTERS

# Genres excluded from matching because they do not suit a workout
GYM_EXCLUDE_GENRES = [
    "honky-tonk",
    "bluegrass",
    "tango",
    "country",
    "folk",
    "gospel",
    "jazz",
    "blues",
    "rockabilly",
    "classical",
    "opera",
]

# Portion of the total playlist duration given to each phase
PHASE_SHARES = {"warmup": 0.2, "peak": 0.6, "cooldown": 0.2}

# Manually written targets in case the LLM call fails
DEFAULT_PLAN = {
    "warmup": {
        "energy": 0.6,
        "valence": 0.6,
        "tempo": 110,
        "danceability": 0.65,
        "loudness": -8,
        "acousticness": 0.2,
        "instrumentalness": 0.05,
    },
    "peak": {
        "energy": 0.9,
        "valence": 0.4,
        "tempo": 135,
        "danceability": 0.55,
        "loudness": -5,
        "acousticness": 0.1,
        "instrumentalness": 0.1,
    },
    "cooldown": {
        "energy": 0.35,
        "valence": 0.4,
        "tempo": 100,
        "danceability": 0.5,
        "loudness": -11,
        "acousticness": 0.6,
        "instrumentalness": 0.05,
    },
}


@dataclass
class Library:
    """The track table, its scaled matrix and the scaler used to scale it"""

    df: pd.DataFrame
    X: np.ndarray
    scaler: StandardScaler


def top_matches(
    lib,
    target,
    clusters,
    n=20,
    pool_size=60,
    min_popularity=55,
    exclude_genres=GYM_EXCLUDE_GENRES,
    include_genres=None,
    seed=None,
):
    """Return n tracks close to a target feature vector.

    Candidates are filtered by cluster, popularity and genre. The pool_size
    closest candidates (Euclidean distance in scaled feature space) are kept,
    and n of them are sampled at random so repeated calls give varied results.
    """
    # The target must be scaled with the same scaler as the tracks
    t = lib.scaler.transform(pd.DataFrame([target])[FEATURES])

    mask = lib.df["cluster"].isin(clusters) & (lib.df["popularity"] >= min_popularity)
    if exclude_genres:
        mask &= ~lib.df["track_genre"].isin(exclude_genres)
    if include_genres:
        mask &= lib.df["track_genre"].isin(include_genres)
    mask = mask.to_numpy()

    out = lib.df[mask].copy()
    out["distance"] = np.linalg.norm(lib.X[mask] - t, axis=1)

    pool = out.sort_values("distance").head(pool_size)
    return pool.sample(n=min(n, len(pool)), random_state=seed).sort_values("distance")


def fill_phase(
    lib,
    target,
    clusters,
    budget_ms,
    used_ids,
    artist_counts,
    max_per_artist=2,
    tolerance_ms=60_000,
    seed=None,
    include_genres=None,
):
    """Select tracks for one phase until its time limit is reached.

    Skips tracks already used, artists that reached max_per_artist, and
    tracks that would extend time limit by more than tolerance_ms.
    used_ids and artist_counts are updated in place so they carry across phases.
    """
    # n equals pool_size so this returns the whole pool of closest matches
    pool = top_matches(
        lib,
        target,
        clusters,
        n=60,
        pool_size=60,
        seed=seed,
        include_genres=include_genres,
    )
    pool = pool.sample(frac=1, random_state=seed)

    chosen, total = [], 0
    for _, row in pool.iterrows():
        if total >= budget_ms:
            break
        artist = row["main_artist"]
        if (
            row["track_id"] in used_ids
            or artist_counts.get(artist, 0) >= max_per_artist
        ):
            continue
        if total + row["duration_ms"] > budget_ms + tolerance_ms:
            continue
        chosen.append(row)
        total += row["duration_ms"]
        used_ids.add(row["track_id"])
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
    return pd.DataFrame(chosen)


def build_playlist(lib, plan, duration_min=45, seed=None, include_genres=None):
    """Build a warmup --> peak --> cooldown playlist"""
    total_ms = duration_min * 60_000
    used_ids, artist_counts = set(), {}

    phases = {}
    for phase, share in PHASE_SHARES.items():
        phases[phase] = fill_phase(
            lib,
            plan[phase],
            PHASE_CLUSTERS[phase],
            share * total_ms,
            used_ids,
            artist_counts,
            seed=seed,
            include_genres=include_genres,
        )

    # Energy rises through the warmup and dips through the cooldown
    phases["warmup"] = phases["warmup"].sort_values("energy")
    phases["cooldown"] = phases["cooldown"].sort_values("energy", ascending=False)

    parts = [df.assign(phase=name) for name, df in phases.items()]
    return pd.concat(parts, ignore_index=True)
