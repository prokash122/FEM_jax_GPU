"""Print configuration for DIW (Direct Ink Writing) continuous printing."""

from dataclasses import dataclass, field


@dataclass
class PrintConfig:
    """All tunable parameters for DIW G-code generation."""

    # Geometry
    layer_height: float = 0.4  # mm
    nozzle_diameter: float = 0.84  # mm
    extrusion_width: float = 0.9  # mm

    # Speeds
    print_speed: float = 10.0  # mm/s
    travel_speed: float = 30.0  # mm/s

    # Extrusion
    extrusion_multiplier: float = 1.0
    filament_diameter: float = 1.75  # mm (for E-value calculation)
    use_pressure_mode: bool = False  # True = pressure-based, no E values
    pressure: float = 0.0  # PSI for pneumatic systems

    # Retraction (typically 0 for DIW)
    retraction_distance: float = 0.0
    z_hop: float = 0.0

    # Temperature (often unused for DIW)
    bed_temp: float = 0.0
    nozzle_temp: float = 0.0

    # Perimeters and infill
    num_perimeters: int = 2
    infill_density: float = 0.5  # 0.0 to 1.0
    infill_pattern: str = "zigzag"  # "zigzag", "spiral", "hilbert"
    infill_angle: float = 45.0  # degrees, rotates per layer
    infill_angle_increment: float = 90.0  # rotation between layers

    # Optimization
    optimize_path: bool = True  # Enable Chinese Postman / Euler optimization

    # Pressure control M-codes (printer-specific)
    pressure_on_code: str = "M750"
    pressure_off_code: str = "M751"
