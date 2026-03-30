"""DIW Continuous Print G-code Generator.

Smart G-code generation for Direct Ink Writing (DIW) printers.
Generates continuous toolpaths from STL files, minimizing travel moves
using Chinese Postman / Eulerian path optimization.
"""

from .config import PrintConfig
from .mesh_loader import load_mesh, prepare_mesh
from .slicer import slice_mesh
from .contour import extract_contours
from .infill import generate_infill
from .toolpath import generate_all_toolpaths
from .gcode_writer import write_gcode

__version__ = "1.0.0"

__all__ = [
    "PrintConfig",
    "load_mesh",
    "prepare_mesh",
    "slice_mesh",
    "extract_contours",
    "generate_infill",
    "generate_all_toolpaths",
    "write_gcode",
]
