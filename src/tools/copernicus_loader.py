"""
Copernicus NetCDF loader for chlorophyll and phytoplankton data.
Loads the copernicus.nc file at startup and provides nearest-point lookup.
"""
import os
import math
import numpy as np

# Path to the NC file
NC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "copernicus.nc")

_ds = None
_chl_data = None
_phyc_data = None
_lats = None
_lons = None

def _load():
    global _ds, _chl_data, _phyc_data, _lats, _lons
    if _chl_data is not None:
        return True
    try:
        import xarray as xr
        _ds = xr.open_dataset(NC_PATH)
        # Squeeze out the time and depth dimensions (both size=1)
        _chl_data = _ds["chl"].isel(time=0, depth=0).values   # shape (lat, lon)
        _phyc_data = _ds["phyc"].isel(time=0, depth=0).values  # shape (lat, lon)
        _lats = _ds["latitude"].values   # 1-D array
        _lons = _ds["longitude"].values  # 1-D array
        print(f"[copernicus_loader] Loaded {NC_PATH}: chl grid {_chl_data.shape}, "
              f"lat=[{_lats.min()},{_lats.max()}], lon=[{_lons.min()},{_lons.max()}]")
        return True
    except Exception as e:
        print(f"[copernicus_loader] Failed to load {NC_PATH}: {e}")
        return False


def get_copernicus_data(lat: float, lon: float) -> dict:
    """
    Return interpolated chlorophyll-a (mg m-3) and phytoplankton (mmol m-3)
    from the Copernicus NetCDF at the nearest grid point to (lat, lon).

    Returns a dict with keys:
        chlorophyll_mg_m3, phytoplankton_mmol_m3, source, grid_lat, grid_lon,
        distance_km, in_coverage (bool)
    """
    if not _load():
        return {
            "chlorophyll_mg_m3": None,
            "phytoplankton_mmol_m3": None,
            "source": "unavailable (copernicus.nc failed to load)",
            "in_coverage": False
        }

    # Find nearest latitude index
    lat_idx = int(np.argmin(np.abs(_lats - lat)))
    lon_idx = int(np.argmin(np.abs(_lons - lon)))

    nearest_lat = float(_lats[lat_idx])
    nearest_lon = float(_lons[lon_idx])

    # Haversine distance from query to grid point
    R = 6371.0
    dlat = math.radians(nearest_lat - lat)
    dlon = math.radians(nearest_lon - lon)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(nearest_lat)) * math.sin(dlon/2)**2
    dist_km = round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a)), 2)

    chl_val = _chl_data[lat_idx, lon_idx]
    phyc_val = _phyc_data[lat_idx, lon_idx]

    # Check if the point is within grid coverage (lat/lon must be inside grid bounds)
    in_coverage = (
        float(_lats.min()) <= lat <= float(_lats.max()) and
        float(_lons.min()) <= lon <= float(_lons.max())
    )

    result = {
        "in_coverage": in_coverage,
        "grid_lat": nearest_lat,
        "grid_lon": nearest_lon,
        "distance_to_grid_km": dist_km,
    }

    if chl_val is not None and not np.isnan(chl_val):
        result["chlorophyll_mg_m3"] = round(float(chl_val), 4)
        result["source"] = "Copernicus Marine Service (CMEMS biogeochemical forecast)"
    else:
        result["chlorophyll_mg_m3"] = None
        result["source"] = "Copernicus Marine Service (CMEMS) — NaN at this grid point (likely land/coastal)"

    if phyc_val is not None and not np.isnan(phyc_val):
        result["phytoplankton_mmol_m3"] = round(float(phyc_val), 4)
    else:
        result["phytoplankton_mmol_m3"] = None

    # PFZ inference: high chlorophyll (>1.5 mg/m3) = productive fishing zone
    if result["chlorophyll_mg_m3"] is not None:
        chl = result["chlorophyll_mg_m3"]
        result["is_pfz"] = chl >= 1.0
        # pfz_classification_confidence refers to bloom strength, NOT data quality
        result["pfz_classification_confidence"] = "high" if chl >= 2.0 else ("medium" if chl >= 1.0 else "low (below PFZ threshold)")
        result["pfz_source"] = "derived from Copernicus chlorophyll-a (PFZ threshold: chl >= 1.0 mg/m3)"
    else:
        result["is_pfz"] = None
        result["pfz_source"] = "unavailable — no chlorophyll data at this point"

    return result


def get_copernicus_bbox(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> list:
    """
    Return all valid grid points within the bounding box.
    Returns a list of dicts, each with lat, lon, chlorophyll_mg_m3, is_pfz etc.
    """
    if not _load():
        return []

    results = []
    for i, lat_v in enumerate(_lats):
        for j, lon_v in enumerate(_lons):
            if min_lat <= float(lat_v) <= max_lat and min_lon <= float(lon_v) <= max_lon:
                chl_val = _chl_data[i, j]
                phyc_val = _phyc_data[i, j]
                if not np.isnan(chl_val):
                    chl = round(float(chl_val), 4)
                    results.append({
                        "lat": float(lat_v),
                        "lon": float(lon_v),
                        "chlorophyll_mg_m3": chl,
                        "phytoplankton_mmol_m3": round(float(phyc_val), 4) if not np.isnan(phyc_val) else None,
                        "is_pfz": chl >= 1.0,
                        "pfz_classification_confidence": "high" if chl >= 2.0 else ("medium" if chl >= 1.0 else "low (below threshold)"),
                        "source": "Copernicus Marine Service (CMEMS)"
                    })
    return results


def get_coverage_bounds() -> dict:
    """Return the lat/lon bounds of the loaded Copernicus data."""
    if not _load():
        return {}
    return {
        "min_lat": float(_lats.min()),
        "max_lat": float(_lats.max()),
        "min_lon": float(_lons.min()),
        "max_lon": float(_lons.max()),
        "date": "2026-09-14"
    }
