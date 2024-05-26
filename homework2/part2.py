import matplotlib.pyplot as plt
import numpy as np

type vector = list[float]


# 1 Create python function that calculates angle between 2 vectors in degrees (use numpy functions to implement the function).
def calculate_angle(a: vector, b: vector) -> float:
    """
    Calculates the angle in degrees between two vectors.

    Parameters:
    a (vector): The first vector.
    b (vector): The second vector.

    Returns:
    float: The angle between the two vectors in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    radians = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    degrees = np.degrees(radians)
    return degrees


a = [1, 0, 0]
b = [0, 1, 0]

c = [1, 2, 3]
d = [3, 2, 1]

f = [1, 0.36, 0]
g = [-1, 0, 0]

# 2. Use your function to calculate angles between the next pairs of vectors:
print(calculate_angle(a, b))
print(calculate_angle(c, d))
print(calculate_angle(f, g))

# 3. Visualize vectors in 2D space.
vectors = [a, b, c, d, f, g]
x_coords = [vector[0] for vector in vectors]
y_coords = [vector[1] for vector in vectors]
origin_x = [0 for vector in vectors]  # x origin coords
origin_y = [0 for vector in vectors]  # y origin coords
plt.quiver(
    origin_x,
    origin_y,
    x_coords,
    y_coords,
    color=["r", "b", "g", "c", "m", "y"],
    scale=10,
)
plt.show()
