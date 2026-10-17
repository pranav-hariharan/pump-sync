"""Convert a text workout description into phase target feature vectors."""

import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from src.data import FEATURES, PHASE_CLUSTERS
from src.matching import DEFAULT_PLAN

load_dotenv()
logger = logging.getLogger(__name__)

# The next model is used when one is overloaded or unavailable.
MODELS = ["gemini-3.5-flash-lite", "gemini-3.7-flash", "gemini-3.8-flash"]
REQUEST_TIMEOUT_MS = 30_000
MAX_INPUT_CHARS = 300


class PhaseTarget(BaseModel):
    energy: float
    valence: float
    tempo: float
    danceability: float
    loudness: float
    acousticness: float
    instrumentalness: float


class MoodPlan(BaseModel):
    warmup: PhaseTarget
    peak: PhaseTarget
    cooldown: PhaseTarget


SYSTEM_PROMPT_TEMPLATE = """You design workout playlists. Given a description of a gym session
and the user's mood, output target audio features for three phases:
warmup (first ~20%), peak (middle ~60%), cooldown (last ~20%).

Feature ranges:
- energy, valence, danceability, acousticness, instrumentalness: 0.0 to 1.0
- tempo: beats per minute
- loudness: decibels, negative (quieter is lower)

The song library only contains songs like these typical profiles (median values):
warmup: {warmup}
peak: {peak}
cooldown: {cooldown}

Start from these profiles and adjust to fit the user's request. Shift energy,
valence, danceability, acousticness and instrumentalness by at most about 0.15,
tempo by at most about 15 BPM, and loudness by at most about 3 dB. Do not go
far outside these profiles, because no songs exist there.
Warmup is moderate, peak is the most intense, cooldown is the calmest.
Cardio suits higher tempo; heavy lifting suits lower tempo with high energy."""


class MoodParser:
    """Turns a workout description into a plan of target vectors using LLM

    The prompt is grounded in the track library: it includes each phase's median
    feature profile, and returned targets are clamped to the 5th-95th percentile
    range of the tracks available for that phase.
    """

    def __init__(self, lib):
        phase_tracks = {
            p: lib.df[lib.df["cluster"].isin(c)] for p, c in PHASE_CLUSTERS.items()
        }
        self.bounds = {
            p: {f: (t[f].quantile(0.05), t[f].quantile(0.95)) for f in FEATURES}
            for p, t in phase_tracks.items()
        }
        profiles = {
            p: t[FEATURES].median().round(2).to_dict() for p, t in phase_tracks.items()
        }
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**profiles)
        self._client = None
        self._cache = {}

    def _get_client(self):
        if self._client is None:
            self._client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )
        return self._client

    def _fit_to_data(self, plan):
        """Clamp each target into the range of tracks that exist for its phase."""
        fitted = {}
        for phase, bounds in self.bounds.items():
            fitted[phase] = {
                f: float(min(max(plan[phase][f], lo), hi))
                for f, (lo, hi) in bounds.items()
            }
        return fitted

    def parse(self, text):
        """Return (plan, used_fallback). Falls back to DEFAULT_PLAN if every call fails."""
        text = text.strip()[:MAX_INPUT_CHARS]
        if not text:
            return DEFAULT_PLAN, True
        if text in self._cache:
            return self._cache[text], False

        for model in MODELS:
            try:
                resp = self._get_client().models.generate_content(
                    model=model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        response_mime_type="application/json",
                        response_schema=MoodPlan,
                        temperature=0.3,
                    ),
                )
                plan = self._fit_to_data(json.loads(resp.text))
            except Exception as e:
                logger.warning("%s failed: %s", model, str(e)[:120])
                time.sleep(2)
                continue
            self._cache[text] = plan
            logger.info("plan generated with %s", model)
            return plan, False

        logger.warning("All models failed, using the default plan")
        return DEFAULT_PLAN, True  # not cached, so the next call retries
