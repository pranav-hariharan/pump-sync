"""PumpSync: describe a workout and get a warmup -> peak -> cooldown playlist."""

import random

import streamlit as st

from src.data import load_all
from src.matching import Library, build_playlist
from src.mood import MoodParser

st.set_page_config(page_title="PumpSync", page_icon="🎧", layout="centered")


@st.cache_resource(show_spinner="Loading the track library...")
def get_resources():
    """Load and cluster the data once per server process."""
    df, X, scaler, _ = load_all()
    lib = Library(df, X, scaler)
    return lib, MoodParser(lib)


lib, parser = get_resources()

st.title("PumpSync")
st.caption("Describe your session and get a playlist shaped to it.")

text = st.text_area(
    "What kind of session is it?",
    placeholder="e.g. feeling sluggish but need to hit a heavy leg day",
    max_chars=300,
)
duration = st.slider("Playlist length (minutes)", 30, 60, 45, step=5)
genres = st.multiselect(
    "Genres (optional)",
    sorted(lib.df["track_genre"].unique()),
    help="Leave empty to allow all genres",
)
if st.button("Generate playlist", type="primary", disabled=not text.strip()):
    with st.spinner("Reading your session and picking tracks..."):
        plan, used_fallback = parser.parse(text)
        playlist = build_playlist(
            lib,
            plan,
            duration,
            seed=random.randrange(10**6),
            include_genres=genres or None,
        )
    if used_fallback:
        st.warning(
            "Couldn't interpret your description, so a default workout "
            "plan was used instead."
        )

    total_min = playlist["duration_ms"].sum() / 60_000
    st.subheader(f"{len(playlist)} tracks, {total_min:.0f} min")

    table = playlist.assign(
        length=(playlist["duration_ms"] // 1000).map(
            lambda s: f"{s // 60}:{s % 60:02d}"
        ),
        link="https://open.spotify.com/track/" + playlist["track_id"],
    )[["phase", "track_name", "artists", "length", "link"]]

    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("Spotify", display_text="Open")
        },
    )

    st.markdown("**Energy across the playlist**")
    st.line_chart(playlist["energy"].reset_index(drop=True))
