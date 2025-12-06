from numpy import cos, sin, exp, abs as np_abs, sqrt, meshgrid, linspace, where
import numpy as np
from .types import NDARRAY, Formula
from typing import Tuple


def gyroid(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    return cos(x) * sin(y) + cos(y) * sin(z) + cos(z) * sin(x)
    

def schwarz_g(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # Schwarz G (Gyroid variant) surface.
    return sin(x) * cos(y) + sin(z) * cos(x) + sin(y) * cos(z)


def l_surface(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # L surface (Lidinoid).
    return 0.5 * (
        sin(2.0 * x) * cos(y) * sin(z)
        + sin(2.0 * y) * cos(z) * sin(x)
        + sin(2.0 * z) * cos(x) * sin(y)
    ) - 0.5 * (
        cos(2.0 * x) * cos(2.0 * y) + cos(2.0 * y) * cos(2.0 * z) + cos(2.0 * z) * cos(2.0 * x)
    )


def schwarz_p(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # Schwarz P (Primitive) surface: -(cos(x) + cos(y) + cos(z)) = 0
    return -(cos(x) + cos(y) + cos(z))


def diamond(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # Diamond structure surface.
    return (
        sin(x) * sin(y) * sin(z)
        + sin(x) * cos(y) * cos(z)
        + cos(x) * sin(y) * cos(z)
        + cos(x) * cos(y) * sin(z)
    )


def holes(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # Holes surface.
    return (cos(x) + cos(y) + cos(z)) + 4 * cos(x) * cos(y) * cos(z)


def lamella(x: NDARRAY, y: NDARRAY, z: NDARRAY) -> NDARRAY:
    # Lamella-like surface (experimental).
    x, y, z = y, z, x
    k = 0.50
    x *= k
    y *= k
    z *= k

    return 0.5 * (
        sin(2.0 * x) * cos(y) * sin(z)
        + sin(2.0 * y) * cos(z) * sin(x)
        + sin(2.0 * z) * cos(x) * sin(y)
    ) - 0.5 * (cos(2.0 * x) * cos(2.0 * z) + cos(2.0 * z) * cos(2.0 * x))