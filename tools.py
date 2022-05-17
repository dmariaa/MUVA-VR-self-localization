import numpy as np

colors = [
    (255, 255, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255)
]

ref_marker_points = np.array([
    [-1, 1, 0, 1],
    [1, 1, 0, 1],
    [1, -1, 0, 1],
    [-1, -1, 0, 1],
    [0, 0, 0, 1]
], dtype=float)

ref_marker_points2 = np.array([
    [-1, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [0, 0, 0]
], dtype=float)

ref_marker_axis = np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
], dtype=float)

margins = np.array([
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
]) * 20

center_point = np.array([0, 0, 0, 1])

# tags matrixes
# 0: 776, 199
# 1: 192, 200
# 2: 484, 689
tags_to_world = {
    0: np.append(np.append(np.identity(3), [[7.76], [1.99], [0]], axis=1), [[0, 0, 0, 1]], axis=0),
    1: np.append(np.append(np.identity(3), [[1.92], [2.00], [0]], axis=1), [[0, 0, 0, 1]], axis=0),
    2: np.append(np.append(np.identity(3), [[4.84], [6.89], [0]], axis=1), [[0, 0, 0, 1]], axis=0)
}


def calculate_corners(K: np.ndarray):
    K_inv = np.linalg.inv(K)
    cx = int(K[0, 2])
    cy = int(K[1, 2])

    corners = K_inv @ np.array([
        [0, 0, 1],
        [cx * 2, 0, 1],
        [cx * 2, cy * 2, 1],
        [0, cy * 2, 1],
    ]).T
    corners = corners.T
    corners = np.append(corners, np.array([[1, 1, 1, 1]]).T, axis=1)
    return corners
