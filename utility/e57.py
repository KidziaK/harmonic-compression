from typing import Tuple

import numpy as np
import pye57 as e57

from pathlib import Path
from numpy.typing import NDArray

def load_e57(path: Path) -> Tuple[NDArray[float], NDArray[int]]:
    e57_file = e57.E57(str(path))
    data = e57_file.read_scan_raw(0)

    points = np.column_stack([
        data['cartesianX'],
        data['cartesianY'],
        data['cartesianZ']
    ])

    if "colorRed" in data:
        colors = np.column_stack([
            data['colorRed'],
            data['colorGreen'],
            data['colorBlue']
        ])
    else:
        colors = None

    return points, colors

def numpy_to_e57(output_path: Path, points: np.ndarray, colors: np.ndarray|None) -> None:
    e57_file = e57.E57(str(output_path), mode="w")

    data = {
        "cartesianX": points[:, 0],  # X coordinates
        "cartesianY": points[:, 1],  # Y coordinates
        "cartesianZ": points[:, 2],  # Z coordinates
    }

    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError("Colors array must have the same shape as points array (N, 3).")

        data["colorRed"] = colors[:, 0].astype(np.uint8)
        data["colorGreen"] = colors[:, 1].astype(np.uint8)
        data["colorBlue"] = colors[:, 2].astype(np.uint8)

    e57_file.write_scan_raw(data)

    e57_file.close()

if __name__ == "__main__":
    example_points = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ]).astype(np.float64)

    numpy_to_e57(Path("../data/cube.e57"), example_points)
