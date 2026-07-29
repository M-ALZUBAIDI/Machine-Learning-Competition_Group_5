"""
Fetches live Near-Earth Object data from NASA's NeoWs API and maps it
to the exact feature format our trained model expects.
"""

import os
import requests

NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
NASA_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"


def fetch_neo_feed(start_date: str, end_date: str) -> dict:
    """
    Calls NASA's NeoWs feed endpoint for a date range (max 7 days).
    Dates must be in 'YYYY-MM-DD' format.
    Returns the raw JSON response.
    """
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "api_key": NASA_API_KEY,
    }

    response = requests.get(NASA_FEED_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def extract_features(asteroid: dict) -> dict:
    """
    Takes one asteroid object from NASA's API response and pulls out
    the 5 features our model was trained on, in the model's units
    (km for diameter/distance, km/h for velocity).
    """
    diameter = asteroid["estimated_diameter"]["kilometers"]

    # Use the first close approach event in the list
    approach = asteroid["close_approach_data"][0]

    return {
        "id": asteroid["id"],
        "name": asteroid["name"],
        "est_diameter_min": diameter["estimated_diameter_min"],
        "est_diameter_max": diameter["estimated_diameter_max"],
        "relative_velocity": float(approach["relative_velocity"]["kilometers_per_hour"]),
        "miss_distance": float(approach["miss_distance"]["kilometers"]),
        "absolute_magnitude": asteroid["absolute_magnitude_h"],
        "close_approach_date": approach["close_approach_date"],
        "nasa_hazardous_flag": asteroid["is_potentially_hazardous_asteroid"],
    }


def get_live_asteroids(start_date: str, end_date: str) -> list[dict]:
    """
    Fetches asteroids for a date range and returns a clean list of
    dicts, one per asteroid, with model-ready features.
    """
    raw = fetch_neo_feed(start_date, end_date)

    asteroids = []
    for date_key, objects in raw["near_earth_objects"].items():
        for obj in objects:
            asteroids.append(extract_features(obj))

    return asteroids
