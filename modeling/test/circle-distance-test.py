import math

def boundary_distance(r,x,y,vx, vy):
    # Calculate the vector from the object to the center of the boundary
    b = 2*(x*vx+y*vy)
    a = (vx**2+vy**2)
    c = (x**2+y**2-r**2)
    return (-b+math.sqrt(b**2-4*a*c))/(2*a)


# Example usage
center_x, center_y = 0, 0
radius = 10
object_x, object_y = 5, 5
velocity_x, velocity_y = -1, -1

distance = boundary_distance(radius, object_x, object_y, velocity_x, velocity_y)
print("Distance to the closest part of the boundary:", distance)
