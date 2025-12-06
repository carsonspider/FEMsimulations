from numpy import mgrid  # type: ignore[attr-defined]
from skimage import measure

from .types import *

def compute_surface(span: float, subdivisions: int, f: Formula, threshold: float = 0.0) -> tuple[NDARRAY, NDARRAY]:
    res = float(subdivisions) * 1j
    mg = mgrid[-1:1:res, -1:1:res, -1:1:res]  # type: ignore
    x, y, z = span * mg
    volume = f(x, y, z)
    vertices, faces, normals, vals = measure.marching_cubes(
        volume,
        threshold,
        spacing=(0.1, 0.1, 0.1),
        allow_degenerate=True,
    )  # type: ignore[no-untyped-call]

    return vertices, faces