import json
import os
import math
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# Get the path to the Desktop/ORCA folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# WMO Weather interpretation codes → human-readable label + lightning risk
# https://open-meteo.com/en/docs
# ---------------------------------------------------------------------------
WMO_THUNDERSTORM_CODES = {95, 96, 99}          # Thunderstorm codes
WMO_HEAVY_RAIN_CODES = {63, 65, 73, 75, 81, 82, 83, 84, 85, 86}
WMO_LABELS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

def _wmo_to_lightning_risk(weather_code: int, wind_gusts_kn: float) -> str:
    """Derive lightning risk from WMO weather code and wind gusts."""
    if weather_code in WMO_THUNDERSTORM_CODES:
        return "High"
    if weather_code in WMO_HEAVY_RAIN_CODES or wind_gusts_kn > 34:
        return "Medium"
    return "Low"

def _is_cyclone_alert(wind_gusts_kn: float, weather_code: int) -> bool:
    """
    Flag cyclone alert if wind gusts exceed tropical storm threshold (34 kn)
    combined with heavy precipitation or thunderstorm codes.
    """
    return wind_gusts_kn >= 34.0 and (weather_code in WMO_THUNDERSTORM_CODES or
                                       weather_code in WMO_HEAVY_RAIN_CODES or
                                       wind_gusts_kn >= 64.0)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class LocationInput(BaseModel):
    location_name: Optional[str] = Field(None, description="Name of the location to look up, e.g. 'Harbor_Point'")
    latitude: Optional[float] = Field(None, description="Latitude of the location")
    longitude: Optional[float] = Field(None, description="Longitude of the location")
    min_lat: Optional[float] = Field(None, description="Southern boundary of a bounding box (minimum latitude)")
    max_lat: Optional[float] = Field(None, description="Northern boundary of a bounding box (maximum latitude)")
    min_lon: Optional[float] = Field(None, description="Western boundary of a bounding box (minimum longitude)")
    max_lon: Optional[float] = Field(None, description="Eastern boundary of a bounding box (maximum longitude)")

def _is_bbox_query(min_lat, max_lat, min_lon, max_lon):
    return all(v is not None for v in [min_lat, max_lat, min_lon, max_lon])

def resolve_location(data: dict, location_name: str = None, lat: float = None, lon: float = None):
    locations = data.get("locations", {})
    if location_name:
        normalized_target = location_name.lower().replace(" ", "").replace("_", "")
        for name, loc_data in locations.items():
            if location_name.lower().replace(" ", "").replace("_", "") == name.lower().replace(" ", "").replace("_", ""):
                return name, loc_data, 0.0
    if lat is not None and lon is not None:
        closest_name, closest_data, min_distance = None, None, float('inf')
        for name, loc_data in locations.items():
            loc_lat, loc_lon = loc_data.get("lat"), loc_data.get("lon")
            if loc_lat is not None and loc_lon is not None:
                dist = haversine_distance(lat, lon, loc_lat, loc_lon)
                if dist < min_distance:
                    min_distance, closest_name, closest_data = dist, name, loc_data
        if closest_name and min_distance <= 50.0:
            return closest_name, closest_data, round(min_distance, 2)
        elif closest_name:
            return None, None, round(min_distance, 2)
    return None, None, None

def resolve_locations_in_bbox(data: dict, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
    locations = data.get("locations", {})
    matches = []
    for name, loc_data in locations.items():
        loc_lat, loc_lon = loc_data.get("lat"), loc_data.get("lon")
        if loc_lat is not None and loc_lon is not None:
            if min_lat <= loc_lat <= max_lat and min_lon <= loc_lon <= max_lon:
                matches.append((name, loc_data))
    return matches

def _fetch_live_weather(lat: float, lon: float) -> dict:
    """
    Fetch live weather from Open-Meteo.
    Returns wind, gusts, weather code, precipitation — everything needed
    to derive lightning risk and cyclone alert without any mock data.
    """
    import requests
    result = {}
    try:
        # Atmosphere: wind, gusts, weather code, precipitation
        atm_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=weather_code,wind_speed_10m,wind_gusts_10m,precipitation"
            f"&wind_speed_unit=kn"
            f"&forecast_days=1"
        )
        atm = requests.get(atm_url, timeout=8).json()
        cur = atm.get("current", {})

        weather_code = cur.get("weather_code", 0) or 0
        wind_speed_kn = cur.get("wind_speed_10m", 0) or 0
        wind_gusts_kn = cur.get("wind_gusts_10m", 0) or 0
        precipitation_mm = cur.get("precipitation", 0) or 0

        result["wind_speed_knots"] = round(wind_speed_kn, 1)
        result["wind_gusts_knots"] = round(wind_gusts_kn, 1)
        result["precipitation_mm"] = round(precipitation_mm, 2)
        result["weather_code"] = weather_code
        result["weather_description"] = WMO_LABELS.get(weather_code, f"Code {weather_code}")
        result["lightning_risk"] = _wmo_to_lightning_risk(weather_code, wind_gusts_kn)
        result["cyclone_alert"] = _is_cyclone_alert(wind_gusts_kn, weather_code)
        result["wind_source"] = "live, Open-Meteo"
        result["lightning_source"] = "live, derived from WMO weather code"
        result["cyclone_source"] = "live, derived from wind gusts threshold (≥34kn storm + ≥64kn typhoon)"

    except Exception as e:
        result["wind_speed_knots"] = "Error"
        result["wind_gusts_knots"] = "Error"
        result["lightning_risk"] = "Unknown"
        result["cyclone_alert"] = False
        result["error_atm"] = str(e)

    try:
        # Marine: wave height, swell, wave period
        marine_url = (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={lat}&longitude={lon}"
            f"&current=wave_height,wind_wave_height,swell_wave_height,wave_period,sea_surface_temperature"
        )
        marine = requests.get(marine_url, timeout=8).json()
        mcur = marine.get("current", {})
        result["wave_height_meters"] = mcur.get("wave_height")
        result["swell_height_meters"] = mcur.get("swell_wave_height")
        result["wave_period_seconds"] = mcur.get("wave_period")
        result["wave_source"] = "live, Open-Meteo Marine"
    except Exception as e:
        result["wave_height_meters"] = "Error"
        result["error_marine"] = str(e)

    return result


async def get_weather_data(
    location_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None
) -> str:
    """
    Retrieve LIVE wave height, wind speed, lightning risk, and cyclone alerts
    for any coordinate on Earth using Open-Meteo APIs.
    Lightning risk is derived from WMO weather codes (95-99 = thunderstorm).
    Cyclone alert is derived from wind gusts exceeding 34 knots storm threshold.
    """
    is_bbox = _is_bbox_query(min_lat, max_lat, min_lon, max_lon)

    if not location_name and (latitude is None or longitude is None) and not is_bbox:
        return "Error: Must provide either location_name, both latitude and longitude, OR bounding box (min_lat, max_lat, min_lon, max_lon)."

    file_path = os.path.join(BASE_DIR, "mock_weather_data.json")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # ── AREA / BOUNDING BOX QUERY ──
        if is_bbox:
            center_lat = (min_lat + max_lat) / 2.0
            center_lon = (min_lon + max_lon) / 2.0
            live = _fetch_live_weather(center_lat, center_lon)

            result = {
                "query_type": "area",
                "bounding_box": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
                "center_latitude": round(center_lat, 4),
                "center_longitude": round(center_lon, 4),
                "center_wind_speed_knots": live.get("wind_speed_knots"),
                "center_wind_gusts_knots": live.get("wind_gusts_knots"),
                "center_wave_height_meters": live.get("wave_height_meters"),
                "center_swell_height_meters": live.get("swell_height_meters"),
                "center_weather": live.get("weather_description"),
                "center_lightning_risk": live.get("lightning_risk"),
                "center_cyclone_alert": live.get("cyclone_alert"),
                "center_source": "live, Open-Meteo"
            }

            # Reference locations inside the box (for supplementary named-station info)
            bbox_matches = resolve_locations_in_bbox(data, min_lat, max_lat, min_lon, max_lon)
            locations_in_area = []
            for name, loc_data in bbox_matches:
                loc_live = _fetch_live_weather(loc_data.get("lat"), loc_data.get("lon"))
                locations_in_area.append({
                    "name": name,
                    "lat": loc_data.get("lat"),
                    "lon": loc_data.get("lon"),
                    "wind_speed_knots": loc_live.get("wind_speed_knots"),
                    "wind_gusts_knots": loc_live.get("wind_gusts_knots"),
                    "wave_height_meters": loc_live.get("wave_height_meters"),
                    "weather": loc_live.get("weather_description"),
                    "lightning_risk": loc_live.get("lightning_risk"),
                    "cyclone_alert": loc_live.get("cyclone_alert"),
                    "source": "live, Open-Meteo"
                })
            result["locations_in_area"] = locations_in_area
            result["locations_found"] = len(locations_in_area)
            return json.dumps(result)

        # ── SINGLE POINT QUERY ──
        # Try to resolve a named location for geo context (name display)
        matched_name, loc_data, dist_km = resolve_location(data, location_name, latitude, longitude)

        # Use provided coordinates; fall back to matched location coords
        fetch_lat = latitude if latitude is not None else (loc_data.get("lat") if loc_data else None)
        fetch_lon = longitude if longitude is not None else (loc_data.get("lon") if loc_data else None)

        if fetch_lat is None or fetch_lon is None:
            return json.dumps({"error": "Could not determine coordinates for this location."})

        # All weather data is now LIVE — no mock fallback needed
        live = _fetch_live_weather(fetch_lat, fetch_lon)
        result = {
            "resolved_latitude": fetch_lat,
            "resolved_longitude": fetch_lon,
            "matched_location": matched_name,
            "distance_to_reference_km": dist_km,
            **live
        }
        return json.dumps(result)

    except Exception as e:
        return f"Error reading weather data: {str(e)}"

weather_tool = StructuredTool.from_function(
    func=get_weather_data,
    name="get_weather_data",
    description=(
        "Get LIVE wind speed, wave height, lightning risk, and cyclone alerts for any coordinate on Earth. "
        "Lightning risk is derived from real WMO weather codes (thunderstorm = High). "
        "Cyclone alert is derived from wind gusts ≥ 34 knots. "
        "Accepts a single point (latitude/longitude or location_name) or a bounding box area "
        "(min_lat, max_lat, min_lon, max_lon)."
    ),
    args_schema=LocationInput,
    coroutine=get_weather_data
)


def _fetch_live_sst(lat: float, lon: float) -> dict:
    """
    Fetch live Sea Surface Temperature and ocean current velocity from Open-Meteo Marine API.
    """
    import requests
    result = {}
    try:
        url = (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={lat}&longitude={lon}"
            f"&current=sea_surface_temperature,ocean_current_velocity,ocean_current_direction"
        )
        r = requests.get(url, timeout=8).json()
        cur = r.get("current", {})
        result["sst_celsius"] = cur.get("sea_surface_temperature")
        result["ocean_current_velocity_kmh"] = cur.get("ocean_current_velocity")
        result["ocean_current_direction_deg"] = cur.get("ocean_current_direction")
        result["sst_source"] = "live, Open-Meteo Marine"
    except Exception as e:
        result["sst_celsius"] = "Error"
        result["sst_source"] = "live (failed)"
        result["error"] = str(e)
    return result


async def get_marine_data(
    location_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None
) -> str:
    """
    Retrieve LIVE Sea Surface Temperature and ocean current data for any coordinate.
    Chlorophyll-a and PFZ status come from Copernicus Marine Service NetCDF (copernicus.nc).
    """
    from src.tools.copernicus_loader import get_copernicus_data, get_copernicus_bbox

    is_bbox = _is_bbox_query(min_lat, max_lat, min_lon, max_lon)

    if not location_name and (latitude is None or longitude is None) and not is_bbox:
        return "Error: Must provide either location_name, both latitude and longitude, OR bounding box (min_lat, max_lat, min_lon, max_lon)."

    file_path = os.path.join(BASE_DIR, "mock_marine_data.json")
    try:
        with open(file_path, "r") as f:
            ref_data = json.load(f)

        # ── AREA / BOUNDING BOX QUERY ──
        if is_bbox:
            center_lat = (min_lat + max_lat) / 2.0
            center_lon = (min_lon + max_lon) / 2.0
            live = _fetch_live_sst(center_lat, center_lon)

            result = {
                "query_type": "area",
                "bounding_box": {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon},
                "center_latitude": round(center_lat, 4),
                "center_longitude": round(center_lon, 4),
                "center_sst_celsius": live.get("sst_celsius"),
                "center_ocean_current_kmh": live.get("ocean_current_velocity_kmh"),
                "center_source": "live, Open-Meteo Marine",
            }

            # Copernicus grid points inside the box
            copernicus_points = get_copernicus_bbox(min_lat, max_lat, min_lon, max_lon)
            pfz_zones = [p for p in copernicus_points if p.get("is_pfz")]
            non_pfz_zones = [p for p in copernicus_points if not p.get("is_pfz")]

            # For each PFZ zone, add live SST
            enriched_pfz = []
            for p in pfz_zones:
                sst_live = _fetch_live_sst(p["lat"], p["lon"])
                enriched_pfz.append({**p, "sst_celsius": sst_live.get("sst_celsius"), "sst_source": "live, Open-Meteo Marine"})

            result["pfz_zones_in_area"] = enriched_pfz
            result["other_locations_in_area"] = non_pfz_zones
            result["total_pfz_found"] = len(pfz_zones)
            result["total_copernicus_points"] = len(copernicus_points)
            result["chlorophyll_source"] = "Copernicus Marine Service (CMEMS biogeochemical forecast)"

            if not copernicus_points:
                result["note"] = "Selected area is outside the Copernicus data coverage (Gujarat Coast: lat 20-24, lon 68-72). Only live SST for center point is available."

            return json.dumps(result)

        # ── SINGLE POINT QUERY ──
        # Resolve location name if provided
        matched_name, loc_data, dist_km = resolve_location(ref_data, location_name, latitude, longitude)
        fetch_lat = latitude if latitude is not None else (loc_data.get("lat") if loc_data else None)
        fetch_lon = longitude if longitude is not None else (loc_data.get("lon") if loc_data else None)

        if fetch_lat is None or fetch_lon is None:
            return json.dumps({"error": "Could not determine coordinates for this location."})

        # Live SST + ocean current — works for any coordinate
        live = _fetch_live_sst(fetch_lat, fetch_lon)

        # Copernicus chlorophyll — works within Gujarat Coast grid
        cop = get_copernicus_data(fetch_lat, fetch_lon)

        result = {
            "resolved_latitude": fetch_lat,
            "resolved_longitude": fetch_lon,
            "matched_location": matched_name,
            "distance_to_reference_km": dist_km,
            "sst_celsius": live.get("sst_celsius"),
            "ocean_current_velocity_kmh": live.get("ocean_current_velocity_kmh"),
            "ocean_current_direction_deg": live.get("ocean_current_direction_deg"),
            "sst_source": "live, Open-Meteo Marine",
            "chlorophyll_mg_m3": cop.get("chlorophyll_mg_m3"),
            "phytoplankton_mmol_m3": cop.get("phytoplankton_mmol_m3"),
            "chlorophyll_source": cop.get("source"),
            "is_pfz": cop.get("is_pfz"),
            "pfz_confidence": cop.get("pfz_confidence"),
            "pfz_source": cop.get("pfz_source"),
            "copernicus_grid_point": f"{cop.get('grid_lat')}, {cop.get('grid_lon')}",
            "copernicus_distance_km": cop.get("distance_to_grid_km"),
        }

        if not cop.get("in_coverage"):
            result["coverage_note"] = (
                "This location is outside the current Copernicus data grid. "
                "SST and ocean current are still live and accurate."
            )

        return json.dumps(result)

    except Exception as e:
        return f"Error reading marine data: {str(e)}"

marine_tool = StructuredTool.from_function(
    func=get_marine_data,
    name="get_marine_data",
    description=(
        "Get LIVE SST and ocean current data for any coordinate on Earth. "
        "Chlorophyll-a and PFZ status are from the Copernicus Marine Service (CMEMS) NetCDF covering Gujarat Coast. "
        "Accepts a single point (latitude/longitude or location_name) or a bounding box area."
    ),
    args_schema=LocationInput,
    coroutine=get_marine_data
)
