"""Data loading, cleaning, feature scaling and clustering for PumpSync."""

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "songs.csv"

# The lyrics column is skipped: it makes up most of the file size
USE_COLUMNS = [
    "id",
    "name",
    "artists",
    "year",
    "genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "tempo",
    "duration_ms",
]

# Audio features used for clustering and matching
FEATURES = [
    "energy",
    "valence",
    "tempo",
    "danceability",
    "loudness",
    "acousticness",
    "instrumentalness",
]

# Oldest release year kept
MIN_YEAR = 2000

# Tracks below this popularity are almost always obscure uploads or duplicates
MIN_POPULARITY = 20

# Number of clusters; re-select with the elbow and silhouette scores after any data change
N_CLUSTERS = 5

CLUSTER_NAMES = {
    0: "fast-and-heavy",
    1: "driving-mid-tempo",
    2: "groovy-danceable",
    3: "mellow-acoustic",
    4: "electronic-instrumental",
}

# Clusters whose profiles describe each phase. These ground the LLM prompt and
# bound its targets. Track matching can search all clusters (see matching.py).
PHASE_CLUSTERS = {
    "warmup": [2],
    "peak": [0, 1, 4],
    "cooldown": [3],
}

# Slowed, sped-up, nightcore and live recordings are re-uploads or alternate takes
VARIANT_PATTERN = (
    r"slowed|sped up|speed up|nightcore|[-(\[]\s*live\b|\blive (?:at|in|from|on)\b"
)

# Suffixes that mark the same song re-released (remasters, radio edits, etc.)
SUFFIX_PATTERN = r"\s*[-(\[]\s*(\d{4}\s+)?(remaster(ed)?|radio edit|mono|stereo)\b.*$"


def _parse_artists(value):
    """The artists column stores a JSON list as text, e.g. '["A", "B"]'."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return [str(value)]


def load_and_clean(path=DATA_PATH):
    """Load the tracks CSV and return a cleaned DataFrame."""
    df = pd.read_csv(path, usecols=USE_COLUMNS)

    # Rename to the column names the rest of the code uses
    df = df.rename(
        columns={"id": "track_id", "name": "track_name", "genre": "track_genre"}
    )

    df = df.dropna()
    df = df[(df["year"] >= MIN_YEAR) & (df["popularity"] >= MIN_POPULARITY)]
    df = df[~df["track_name"].str.contains(VARIANT_PATTERN, case=False, regex=True)]

    artist_lists = df["artists"].map(_parse_artists)
    df["main_artist"] = artist_lists.str[0]
    df["artists"] = artist_lists.map(", ".join)

    # The same song appears under several IDs (remasters, re-releases) with
    # unreliable popularity. Keep the most popular version of each title + main artist.
    df["title_key"] = (
        df["track_name"]
        .str.replace(SUFFIX_PATTERN, "", regex=True, case=False)
        .str.lower()
        .str.strip()
    )
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["title_key", "main_artist"])

    df = df[
        (df["tempo"] > 0)  # tempo of 0 is invalid data
        & (df["duration_ms"].between(120_000, 420_000))  # 2-7 minutes
        & (df["speechiness"] < 0.66)  # exclude spoken-word tracks
    ]

    return df.drop(columns="title_key").reset_index(drop=True)


def build_model(df):
    """Standardize the features and cluster the tracks with K-Means.

    Returns the DataFrame with `cluster` and `cluster_name` columns, the
    scaled feature matrix X, the fitted scaler and the fitted KMeans model.
    The scaler is returned so that target vectors can later be transformed
    into the same feature space as the tracks.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
    df = df.copy()
    df["cluster"] = km.fit_predict(X)
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)
    return df, X, scaler, km


def check_clusters(df):
    """Raise an AssertionError if the phase-to-cluster mapping is out of order.

    Cluster numbers are arbitrary, so this checks the property that matters:
    average energy must rise from cooldown to warmup to peak.
    """
    energy = {
        p: df[df["cluster"].isin(c)]["energy"].mean() for p, c in PHASE_CLUSTERS.items()
    }
    assert energy["peak"] > energy["warmup"] > energy["cooldown"], (
        f"phase clusters are not ordered by energy: {energy}"
    )


def load_all(check=True):
    """Load, clean, scale and cluster the data. Returns (df, X, scaler, km)."""
    df = load_and_clean()
    df, X, scaler, km = build_model(df)
    if check:
        check_clusters(df)
    return df, X, scaler, km
