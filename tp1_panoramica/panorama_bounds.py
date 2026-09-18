"""Compatibilidad para importar la geometría del canvas panorámico."""

try:
    from .panorama.geometry import compute_panorama_bounds, transform_corners
except ImportError:
    from panorama.geometry import compute_panorama_bounds, transform_corners

__all__ = ["compute_panorama_bounds", "transform_corners"]
