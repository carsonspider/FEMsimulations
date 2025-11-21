from pathlib import Path

from active_gyroid_gen import (
    DEFAULT_PARAMS,
    GyroidParameters,
    batch_generate_gyroids,
)
from dataclasses import replace


def build_parameter_sets() -> list[GyroidParameters]:
    """Return a small sweep of gyroid parameter variants for quick testing."""
    variants: list[GyroidParameters] = []
    for i in range(1, 6):
        scale = 1.0 + i * 0.05
        wall_thickness = max(0.01, i / 30.0)
        variants.append(
            replace(
                DEFAULT_PARAMS,
                unit_cell_size=DEFAULT_PARAMS.unit_cell_size * scale,
                nsteps=DEFAULT_PARAMS.nsteps,
                porosity_min=0.3,
                porosity_max=0.7,
                grad=1,
                wall_thickness=wall_thickness,
            )
        )
    return variants


def main():
    output_root = Path.cwd() / "gyroid_variants"
    param_sets = build_parameter_sets()
    batch_generate_gyroids(param_sets, output_root)


if __name__ == "__main__":
    main()
   
