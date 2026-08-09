"""
Client for the National Weather Service (NWS) API.

No API key required. NWS asks for a descriptive User-Agent.
Docs: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import requests

logger = logging.getLogger("weather-client")

_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30

# Fixed city → (lat, lon, state) map for homework demos.
# Keys are matched case-insensitively after stripping whitespace.
CITY_COORDS: dict[str, tuple[float, float, str]] = {
    "Chicago, IL": (41.8781, -87.6298, "IL"),
    "Austin, TX": (30.2672, -97.7431, "TX"),
    "New York, NY": (40.7128, -74.0060, "NY"),
    "Seattle, WA": (47.6062, -122.3321, "WA"),
    "Miami, FL": (25.7617, -80.1918, "FL"),
    "Denver, CO": (39.7392, -104.9903, "CO"),
    "San Francisco, CA": (37.7749, -122.4194, "CA"),
    "Boston, MA": (42.3601, -71.0589, "MA"),
    "Los Angeles, CA": (34.0522, -118.2437, "CA"),
    "Atlanta, GA": (33.7490, -84.3880, "GA"),
}

_CITY_LOOKUP = {k.lower(): (k, v) for k, v in CITY_COORDS.items()}


def resolve_location(location: str) -> tuple[str, float, float, str]:
    """
    Resolve a city string to (canonical_name, lat, lon, state).

    Raises ValueError if the city is not in the fixed map.
    """
    key = (location or "").strip().lower()
    if key not in _CITY_LOOKUP:
        known = ", ".join(sorted(CITY_COORDS))
        raise ValueError(
            f"Unknown location {location!r}. Use one of: {known}"
        )
    canonical, (lat, lon, state) = _CITY_LOOKUP[key]
    return canonical, lat, lon, state


def list_known_locations() -> list[str]:
    return sorted(CITY_COORDS.keys())


class WeatherClient:
    """Thin wrapper around api.weather.gov for alerts + forecast narratives."""

    def __init__(
        self,
        base_url: str = _BASE_URL,
        timeout: int = _DEFAULT_TIMEOUT,
        user_agent: str = "databricks-lakebase-weather-homework (weather-retrieval)",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_point(self, lat: float, lon: float) -> dict:
        """Resolve lat/lon to NWS grid metadata (office, gridX, gridY, forecast URL)."""
        return self.get(f"/points/{lat},{lon}")

    def get_active_alerts(self, state: str) -> list[dict]:
        """Return active alert Feature objects for a two-letter state code."""
        data = self.get("/alerts/active", params={"area": state.upper()})
        return data.get("features", []) or []

    def get_forecast(self, office: str, grid_x: int, grid_y: int) -> list[dict]:
        """Return forecast period dicts (each has name, detailedForecast, startTime, ...)."""
        data = self.get(f"/gridpoints/{office}/{grid_x},{grid_y}/forecast")
        periods = (data.get("properties") or {}).get("periods") or []
        return periods

    def harvest_location(self, location: str, limit: int = 50) -> list[dict]:
        """
        Fetch alerts + forecast narratives for one location and normalize
        into weather_documents-shaped records.
        """
        canonical, lat, lon, state = resolve_location(location)
        docs: list[dict] = []

        # --- Alerts (by state) ---
        try:
            alerts = self.get_active_alerts(state)
        except requests.HTTPError as err:
            logger.warning("Failed to fetch alerts for %s (%s): %s", canonical, state, err)
            alerts = []

        for feature in alerts:
            doc = self._normalize_alert(feature, canonical)
            if doc and doc.get("narrative_text"):
                docs.append(doc)
            if len(docs) >= limit:
                return docs[:limit]

        # --- Forecast periods (via grid point) ---
        try:
            point = self.get_point(lat, lon)
            props = point.get("properties") or {}
            office = props.get("gridId") or props.get("cwa")
            grid_x = props.get("gridX")
            grid_y = props.get("gridY")
            if office is None or grid_x is None or grid_y is None:
                raise ValueError(f"Incomplete grid metadata for {canonical}: {props}")
            periods = self.get_forecast(office, int(grid_x), int(grid_y))
        except (requests.HTTPError, ValueError, TypeError) as err:
            logger.warning("Failed to fetch forecast for %s: %s", canonical, err)
            periods = []

        for period in periods:
            if len(docs) >= limit:
                break
            doc = self._normalize_forecast(period, canonical)
            if doc and doc.get("narrative_text"):
                docs.append(doc)

        return docs[:limit]

    def harvest_locations(self, locations: list[str], limit: int = 50) -> list[dict]:
        """Harvest documents for many locations (limit is per location)."""
        all_docs: list[dict] = []
        seen_ids: set[str] = set()
        for loc in locations:
            for doc in self.harvest_location(loc, limit=limit):
                doc_id = doc["id"]
                if doc_id in seen_ids:
                    continue
                seen_ids.add(doc_id)
                all_docs.append(doc)
        return all_docs

    @staticmethod
    def _normalize_alert(feature: dict, location: str) -> dict | None:
        props = feature.get("properties") or {}
        alert_id = props.get("id") or feature.get("id")
        if not alert_id:
            return None

        # Prefer stable NWS id path segment when present
        if isinstance(alert_id, str) and "/" in alert_id:
            alert_id = alert_id.rstrip("/").split("/")[-1]

        headline = props.get("headline") or props.get("event") or "Weather Alert"
        event = props.get("event") or headline
        description = (props.get("description") or "").strip()
        instruction = (props.get("instruction") or "").strip()
        narrative_parts = [p for p in (description, instruction) if p]
        narrative_text = "\n\n".join(narrative_parts).strip()
        if not narrative_text:
            return None

        return {
            "id": f"alert:{alert_id}",
            "location": location,
            "source_type": "alert",
            "headline": headline,
            "event": event,
            "narrative_text": narrative_text,
            "issued_at": props.get("sent") or props.get("effective"),
            "effective_at": props.get("effective") or props.get("onset"),
            "payload": feature,
        }

    @staticmethod
    def _normalize_forecast(period: dict, location: str) -> dict | None:
        detailed = (period.get("detailedForecast") or "").strip()
        if not detailed:
            return None

        name = period.get("name") or "Forecast"
        start = period.get("startTime") or ""
        # Stable dedup key: location + period start + name
        raw_key = f"{location}|{start}|{name}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        doc_id = f"forecast:{digest}"

        return {
            "id": doc_id,
            "location": location,
            "source_type": "forecast",
            "headline": f"{location} — {name}",
            "event": name,
            "narrative_text": detailed,
            "issued_at": start or None,
            "effective_at": start or None,
            "payload": period,
        }
