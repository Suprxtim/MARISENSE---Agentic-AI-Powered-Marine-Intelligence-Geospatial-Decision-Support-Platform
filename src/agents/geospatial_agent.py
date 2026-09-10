import json
import os
from shapely.geometry import Point, shape, box as shapely_box
from pyproj import Geod
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# Load data once at startup
GEOJSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mock_geofence_data.geojson")
geofence_features = []

if os.path.exists(GEOJSON_PATH):
    with open(GEOJSON_PATH, "r") as f:
        data = json.load(f)
        for feature in data.get("features", []):
            geom = shape(feature["geometry"])
            geofence_features.append({
                "geometry": geom,
                "zone_type": feature["properties"].get("zone_type"),
                "zone_name": feature["properties"].get("zone_name")
            })

geod = Geod(ellps="WGS84")

def calculate_distance_km(point1: Point, geom2) -> float:
    """Calculate shortest distance in km between a point and a geometry."""
    # Find the nearest point on geom2 to point1
    # shapely distance is Euclidean. For WGS84, we approximate using pyproj on the nearest coordinates
    from shapely.ops import nearest_points
    p1, p2 = nearest_points(point1, geom2)
    _, _, distance_m = geod.inv(p1.x, p1.y, p2.x, p2.y)
    return distance_m / 1000.0

@tool
def check_geofence(
    location_name: str = None,
    latitude: float = None,
    longitude: float = None,
    min_lat: float = None,
    max_lat: float = None,
    min_lon: float = None,
    max_lon: float = None
) -> str:
    """
    Checks if a given location or area intersects with any restricted marine zones (MPA, IMBL, restricted_waters).
    For a single point: ALWAYS provide latitude and longitude for real-world cities. Do NOT use location_name for cities.
    For an area: provide min_lat, max_lat, min_lon, max_lon as a bounding box.
    Returns a JSON string containing the structured result.
    """
    is_bbox = all(v is not None for v in [min_lat, max_lat, min_lon, max_lon])

    # Resolve location name to coordinates for point queries
    if not is_bbox and location_name and (latitude is None or longitude is None):
        from src.tools.data_tools import resolve_location
        marine_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "mock_marine_data.json")
        try:
            with open(marine_path, "r") as f:
                data = json.load(f)
                _, loc_data, _ = resolve_location(data, location_name)
                if loc_data:
                    latitude = loc_data.get("lat")
                    longitude = loc_data.get("lon")
        except Exception as e:
            print("Error resolving location in geofence:", e)

    # ── AREA / BOUNDING BOX QUERY ──
    if is_bbox:
        bbox_geom = shapely_box(min_lon, min_lat, max_lon, max_lat)
        intersecting_zones = []

        for f in geofence_features:
            geom = f["geometry"]
            if geom.intersects(bbox_geom):
                # Calculate how much of the bbox overlaps
                try:
                    intersection = geom.intersection(bbox_geom)
                    overlap_area = intersection.area if hasattr(intersection, 'area') else 0
                except Exception:
                    overlap_area = 0

                intersecting_zones.append({
                    "zone_type": f["zone_type"],
                    "zone_name": f["zone_name"],
                    "intersects": True,
                    "overlap_area_deg2": round(overlap_area, 6) if overlap_area else 0
                })

        result = {
            "query_type": "area",
            "bounding_box": {
                "min_lat": min_lat, "max_lat": max_lat,
                "min_lon": min_lon, "max_lon": max_lon
            },
            "intersecting_zones": intersecting_zones,
            "total_restricted_zones_found": len(intersecting_zones),
            "has_restricted_zones": len(intersecting_zones) > 0
        }

        if intersecting_zones:
            zone_names = [z["zone_name"] for z in intersecting_zones]
            result["warning"] = f"The selected area overlaps with {len(intersecting_zones)} restricted zone(s): {', '.join(zone_names)}"
        else:
            result["note"] = "No restricted zones found within the selected area."

        return json.dumps(result)

    # ── SINGLE POINT QUERY (existing logic) ──
    if latitude is None or longitude is None:
        return json.dumps({
            "inside_restricted_zone": False,
            "zone_type": None,
            "zone_name": None,
            "distance_to_nearest_boundary_km": 999.9,
            "nearest_zone_type": None,
            "error": "Could not resolve coordinates for the location."
        })

    point = Point(longitude, latitude)
    
    inside_zone = False
    zone_type = None
    zone_name = None
    min_dist_km = float('inf')
    nearest_type = None

    for f in geofence_features:
        geom = f["geometry"]
        # Check intersection
        if geom.contains(point) or geom.intersects(point):
            inside_zone = True
            zone_type = f["zone_type"]
            zone_name = f["zone_name"]
            min_dist_km = 0.0
            nearest_type = zone_type
            break
            
        # If not inside, calculate distance to this feature
        dist = calculate_distance_km(point, geom)
        if dist < min_dist_km:
            min_dist_km = dist
            nearest_type = f["zone_type"]

    result = {
        "inside_restricted_zone": inside_zone,
        "zone_type": zone_type,
        "zone_name": zone_name,
        "distance_to_nearest_boundary_km": round(min_dist_km, 2) if min_dist_km != float('inf') else 999.9,
        "nearest_zone_type": nearest_type,
        "resolved_latitude": latitude,
        "resolved_longitude": longitude
    }
    
    return json.dumps(result)

def get_geospatial_agent():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=800).with_config(tags=["geospatial_llm"])
    
    prompt = """You are the Geospatial/Geofencing Agent. Your job is to check if the user's location or area is inside or near any restricted marine zones (like MPAs or the IMBL). 
Call your check_geofence tool with the coordinates extracted from the conversation.
If the user specifies a bounding box area (min_lat, max_lat, min_lon, max_lon), pass those parameters to the tool.
If the user specifies a single point (latitude, longitude), pass those instead.
When the tool returns its JSON, YOUR FINAL RESPONSE MUST BE THE EXACT, UNMODIFIED JSON STRING. 
CRITICAL: Do NOT truncate strings. Do NOT add spaces. Do NOT shorten names like "Gulf of Kutch Marine Sanctuary". Output the exact JSON directly."""
    
    return create_react_agent(
        llm,
        tools=[check_geofence],
        prompt=prompt
    )
