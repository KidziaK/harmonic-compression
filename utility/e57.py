import numpy as np
import pye57 as e57

from pathlib import Path
from numpy.typing import NDArray

def load_e57(path: Path) -> NDArray[float]:
    e57_file = e57.E57(str(path))
    data = e57_file.read_scan_raw(0)

    points = np.column_stack([
        data['cartesianX'],
        data['cartesianY'],
        data['cartesianZ']
    ])

    return points

def numpy_to_e57(output_path: Path, points: np.ndarray) -> None:
    e57_file = e57.E57(str(output_path), mode="w")

    data = {
        "cartesianX": points[:, 0],  # X coordinates
        "cartesianY": points[:, 1],  # Y coordinates
        "cartesianZ": points[:, 2],  # Z coordinates
    }

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
