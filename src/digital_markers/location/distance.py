"""Geospatial distance utilities."""

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Return great-circle distances in meters for scalar or array coordinates."""
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(lat1),
        np.radians(lon1),
        np.radians(lat2),
        np.radians(lon2),
    )
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
