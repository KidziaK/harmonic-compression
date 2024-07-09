import numpy as np

from numba import njit
from scipy.special import iv, lpmv, factorial

# https://mathworld.wolfram.com/SphericalCoordinates.html
@njit
def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    result = np.zeros_like(points)
    for i in range(points.shape[0]):
        point = points[i]
        x, y, z = point[0], point[1], point[2]
        r = np.sqrt(np.dot(point, point))
        if np.isclose(r, 0):
            result[i] = np.array([0, 0, 0])
            continue
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        result[i] = np.array([r, theta, phi])
    return result

def harmonic_f(l: int, kappa: np.ndarray) -> np.ndarray:
    """ 
    Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
    Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
    """
    return np.sqrt((2 * l + 1) / 4 / np.pi) * iv(l + 0.5, kappa) / iv(0.5, kappa)

# https://en.wikipedia.org/wiki/Wigner_D-matrix
# https://www.cambridge.org/core/books/hilbert-space-methods-in-signal-processing/BA54ECB490D53FF8CB176CFDCE34A962
def wigner_d_matrix(l: int, m: int, beta: np.ndarray) -> np.ndarray:
    return np.sqrt(factorial(l - m) / factorial(l + m)) * lpmv(m, l, np.cos(beta))

def wigner_d_function(l: int, m: int, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.exp(-1.0j * alpha * m) * wigner_d_matrix(l, m, beta)

def compress(point_cloud: np.ndarray, l_max: int) -> np.ndarray:
    spherical = cartesian_to_spherical(point_cloud)
    num_of_harmonics = l_max * (1 + l_max) // 2
    coefficients = np.zeros(num_of_harmonics, dtype=np.complex128)

    kappa = spherical[:, 0]
    theta = spherical[:, 1]
    phi = spherical[:, 2]
    i = 0
    for l in range(l_max):
        for m in range(l):
            i += 1
            coefficients[i] = np.mean(harmonic_f(l, kappa) * wigner_d_function(l, m, theta, phi))

    return coefficients

def test_cartesian_to_spherical():
    cartesian = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [-1, -1, -1]
    ]).astype(np.float64)

    spherical = cartesian_to_spherical(cartesian)

    pi_half = np.pi / 2

    expected = np.array([
        [1, 0, pi_half],
        [1, pi_half, pi_half],
        [1, 0, 0],
        [np.sqrt(3), np.pi / 4, np.arccos(1 / np.sqrt(3))],
        [np.sqrt(3), -np.deg2rad(135), np.arccos(-1 / np.sqrt(3))]
    ]).astype(np.float64)

    assert np.allclose(spherical, expected)
