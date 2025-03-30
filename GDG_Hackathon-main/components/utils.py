import numpy as np


def intersection_numpy(ray_vector, rectangle_vector):
    """
    Finds the intersection point of two lines using NumPy. This is generally faster
    than the pure Python implementation, especially for many calculations.

    Args:
        line1: A tuple of two points, where each point is a tuple (x, y).  e.g., ((x1, y1), (x2, y2))
        line2: A tuple of two points, where each point is a tuple (x, y).  e.g., ((x3, y3), (x4, y4))

    Returns:
        A NumPy array [x, y] representing the intersection point, or None if the lines are parallel or coincident.
    """
    p1, p2 = np.array(ray_vector)
    p3, p4 = np.array(rectangle_vector)

    # Construct the matrix and vector for solving the system of equations.
    A = np.array([p2 - p1, p3 - p4]).T
    b = p3 - p1

    # Check for parallelism/coincidence.
    if np.linalg.det(A) == 0:
        # Check for coincidence using cross-product (generalized for 2D)
        if np.cross(p2 - p1, p3 - p1) == 0:
            return None  # coincident
        else:
            return None  # Parallel

    # Solve for the parameters t and s
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:  # Handle singular matrix (shouldn't happen if we checked det, but better to check)
        return None

    t, s = x[0], x[1]

    # Check if the intersection is within the line segments.
    if 0 <= t <= 1 and 0 <= s <= 1:
        return p1 + t * (p2 - p1)
    else:
        return None


def distance_between_points(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def find_hit_point_on_rectangle(start, end, rect):
    # Do not modify rect.x and rect.y!
    top_left = (rect.x, rect.y)
    top_right = (rect.x + rect.width, rect.y)
    bottom_left = (rect.x, rect.y + rect.height)
    bottom_right = (rect.x + rect.width, rect.y + rect.height)

    last_hit_point = None

    # Define the four edges of the rectangle.
    edges = [
        (top_left, top_right),  # Top edge
        (top_right, bottom_right),  # Right edge
        (bottom_right, bottom_left),  # Bottom edge
        (bottom_left, top_left)  # Left edge
    ]

    for edge in edges:
        p = intersection_numpy((start, end), edge)
        if p is not None:
            p = (p[0], p[1])
            if last_hit_point is None or distance_between_points(start, p) < distance_between_points(start,
                                                                                                     last_hit_point):
                last_hit_point = p

    return last_hit_point



