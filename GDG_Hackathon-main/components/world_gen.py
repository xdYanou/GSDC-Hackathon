import random
from components.obstacle import Obstacle
import math


def spawn_objects(world_boundaries, max_object_size, min_object_size, num_objects):
    """
    Spawn a number of objects with random sizes and positions, keeping corners empty
    :param world_boundaries: The boundaries of the world (left, top, right, bottom)
    :param max_object_size: The maximum size of the object (width, height)
    :param min_object_size: The minimum size of the object (width, height)
    :param num_objects: The number of objects to spawn
    :return: A list of objects
    """
    CORNER_RADIUS = 150  # Clear radius from corners
    objects = []

    # Define corners
    corners = [
        (world_boundaries[0], world_boundaries[1]),  # Top-left
        (world_boundaries[2], world_boundaries[1]),  # Top-right
        (world_boundaries[0], world_boundaries[3]),  # Bottom-left
        (world_boundaries[2], world_boundaries[3])  # Bottom-right
    ]

    def check_corner_distance(x, y, width, height):
        """
        Check if any part of an object would be within the corner radius
        """
        object_corners = [
            (x, y),  # Top-left
            (x + width, y),  # Top-right
            (x, y + height),  # Bottom-left
            (x + width, y + height)  # Bottom-right
        ]

        # Also check the midpoints of each edge to prevent large objects
        # from cutting through corner areas
        edge_midpoints = [
            (x + width / 2, y),  # Top edge
            (x + width / 2, y + height),  # Bottom edge
            (x, y + height / 2),  # Left edge
            (x + width, y + height / 2)  # Right edge
        ]

        check_points = object_corners + edge_midpoints

        for corner_x, corner_y in corners:
            for point_x, point_y in check_points:
                distance = math.sqrt((point_x - corner_x) ** 2 + (point_y - corner_y) ** 2)
                if distance < CORNER_RADIUS:
                    return False
        return True

    attempts = 0
    max_attempts = num_objects * 100  # Prevent infinite loop

    while len(objects) < num_objects and attempts < max_attempts:
        # Generate random size
        width = random.randint(min_object_size[0], max_object_size[0])
        height = random.randint(min_object_size[1], max_object_size[1])

        # Generate random position, keeping some padding from boundaries
        x = random.randint(
            world_boundaries[0],
            world_boundaries[2] - width
        )
        y = random.randint(
            world_boundaries[1],
            world_boundaries[3] - height
        )

        # Check if the object is clear of corner areas
        if check_corner_distance(x, y, width, height):
            # Check for overlap with existing objects
            new_object = Obstacle((x, y), (width, height))

            # Add object if it doesn't overlap with any existing objects
            overlap = False
            for obj in objects:
                if new_object.rect.colliderect(obj.rect):
                    overlap = True
                    break

            if not overlap:
                objects.append(new_object)

        attempts += 1

    if attempts >= max_attempts:
        print(f"Warning: Could only place {len(objects)} objects out of {num_objects} requested")

    return objects