"""
Fetches live close-approach asteroid data from JPL's SBDB Close-Approach
Data (CAD) API and maps it to the exact feature format our trained model
expects.

We use JPL's CAD API (ssd-api.jpl.nasa.gov) instead of NASA's NeoWs feed
(api.nasa.gov) because the api.nasa.gov gateway has been unreliable
(intermittent 503s). JPL's CAD API serves the same underlying close-approach
data directly from JPL, requires no API key, and has been stable in testing.
"""

import math
import requests

JPL_CAD_URL = "https://ssd-api.jpl.nasa.gov/cad.api"

AU_TO_KM = 149_597_870.7   # 1 astronomical unit in km
KMS_TO_KMH = 3600          # km/s -> km/h

# Same diameter-from-magnitude formula NASA's own NeoWs feed uses:
# D (km) = 1329 / sqrt(albedo) * 10^(-H/5)
# min uses the high-albedo bound (0.25), max uses the low-albedo bound (0.05)
ALBEDO_MIN_BOUND = 0.25
ALBEDO_MAX_BOUND = 0.05


def estimate_diameter_km(h: float) -> tuple[float, float]:
    """Estimate (min, max) diameter in km from absolute magnitude H."""
    diameter_min = 1329 / math.sqrt(ALBEDO_MIN_BOUND) * (10 ** (-h / 5))
    diameter_max = 1329 / math.sqrt(ALBEDO_MAX_BOUND) * (10 ** (-h / 5))
    return diameter_min, diameter_max


def fetch_close_approach_data(start_date: str, end_date: str, dist_max_au: float = 0.5) -> dict:
    """
    Calls JPL's CAD API for a date range. Dates must be 'YYYY-MM-DD'.
    dist_max_au widens the search beyond CNEOS's tight 0.05 au default
    so we get a reasonable number of asteroids to predict on.
    Returns the raw JSON response.
    """
    params = {
        "date-min": start_date,
        "date-max": end_date,
        "dist-max": dist_max_au,
        "diameter": "true",
        "fullname": "true",
        "sort": "date",
    }

    response = requests.get(JPL_CAD_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def extract_features(row: dict) -> dict:
    """
    Takes one close-approach record (already zipped into a dict using
    the API's 'fields' list) and pulls out the 5 features our model
    was trained on, converted into the same units as the training data
    (km for diameter/distance, km/h for velocity).
    """
    h = float(row["h"])

    # Use JPL's measured diameter if available, otherwise estimate from H
    if row.get("diameter"):
        diameter_km = float(row["diameter"])
        est_diameter_min = diameter_km
        est_diameter_max = diameter_km
    else:
        est_diameter_min, est_diameter_max = estimate_diameter_km(h)

    return {
        "id": row["des"],
        "name": row.get("fullname", row["des"]).strip(),
        "est_diameter_min": est_diameter_min,
        "est_diameter_max": est_diameter_max,
        "relative_velocity": float(row["v_rel"]) * KMS_TO_KMH,
        "miss_distance": float(row["dist"]) * AU_TO_KM,
        "absolute_magnitude": h,
        "close_approach_date": row["cd"],
    }


def get_live_asteroids(start_date: str, end_date: str) -> list[dict]:
    """
    Fetches asteroids for a date range from JPL's CAD API and returns
    a clean list of dicts, one per asteroid, with model-ready features.
    Skips any record that's missing a required field instead of
    crashing the whole request.
    """
    raw = fetch_close_approach_data(start_date, end_date)

    fields = raw["fields"]
    rows = raw.get("data", [])

    asteroids = []
    skipped = 0
    for values in rows:
        record = dict(zip(fields, values))
        try:
            asteroids.append(extract_features(record))
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue

    if skipped:
        print(f"Skipped {skipped} record(s) with incomplete data")

    return asteroids
