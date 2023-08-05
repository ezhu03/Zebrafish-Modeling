import math

def distance_to_closest_part_of_boundary(center_x, center_y, radius, object_x, object_y, velocity_x, velocity_y):
    # Calculate the vector from the object to the center of the boundary
    dx = center_x - object_x
    dy = center_y - object_y

    # Calculate the dot product of the position vector and velocity vector
    dot_product = dx * velocity_x + dy * velocity_y

    # Check if the object is moving toward the boundary
    if dot_product > 0:
        # Calculate the squared magnitude of the position vector
        magnitude_squared = dx ** 2 + dy ** 2

        # Check if the object is inside the boundary
        if magnitude_squared < radius ** 2:
            # Calculate the squared distance from the object to the boundary
            distance_squared = magnitude_squared - (dot_product ** 2) / (velocity_x ** 2 + velocity_y ** 2)

            # Check if the object will intersect with the boundary
            if distance_squared <= radius ** 2:
                # Calculate the distance to the intersection point
                distance_to_intersection = math.sqrt(distance_squared)
                return distance_to_intersection
            else:
                # The object will not intersect with the boundary
                return float('inf')
        else:
            # The object is already outside the boundary
            return 0
    else:
        # The object is moving away from the boundary or parallel to it
        return float('inf')

# Example usage
center_x, center_y = 0, 0
radius = 10
object_x, object_y = 5, 5
velocity_x, velocity_y = -1, -1

distance = distance_to_closest_part_of_boundary(center_x, center_y, radius, object_x, object_y, velocity_x, velocity_y)
print("Distance to the closest part of the boundary:", distance)
