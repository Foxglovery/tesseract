from vpython import *
import math

# Set up the scene
scene.background = color.black
scene.width = 800
scene.height = 600
scene.title = "Sphere of Light Points"

# Parameters for the points
default_radius = 0.05
hover_radius = 0.1
default_color = vector(0.5, 0.5, 0.5)  # dim white
hover_color = vector(1, 1, 1)            # bright white

# Create grid-style points on a sphere using latitude/longitude
points = []
n_lat = 10   # number of latitude divisions (excluding poles)
n_lon = 20   # number of longitude divisions

# Create points for the latitudes (avoid duplicating the poles)
for i in range(1, n_lat):
    lat = math.pi * i / n_lat  # from 0 to pi
    for j in range(n_lon):
        lon = 2 * math.pi * j / n_lon
        x = math.sin(lat) * math.cos(lon)
        y = math.cos(lat)
        z = math.sin(lat) * math.sin(lon)
        pt = sphere(pos=vector(x, y, z),
                    radius=default_radius,
                    color=default_color,
                    emissive=True)
        pt.default_radius = default_radius
        pt.default_color = default_color
        points.append(pt)

# Add the poles
north = sphere(pos=vector(0, 1, 0), radius=default_radius,
               color=default_color, emissive=True)
north.default_radius = default_radius
north.default_color = default_color
points.append(north)

south = sphere(pos=vector(0, -1, 0), radius=default_radius,
               color=default_color, emissive=True)
south.default_radius = default_radius
south.default_color = default_color
points.append(south)

# Variables for mouse-driven rotation
drag = False
prev_mouse_pos = None
rotation_sensitivity = 0.005

def rotate_points(angle, axis):
    """Rotate all points about the origin."""
    for pt in points:
        pt.pos = rotate(pt.pos, angle=angle, axis=axis)

def on_mouse_down(evt):
    global drag, prev_mouse_pos
    drag = True
    prev_mouse_pos = scene.mouse.pos

def on_mouse_up(evt):
    global drag, prev_mouse_pos
    drag = False
    prev_mouse_pos = None

def on_mouse_move(evt):
    global prev_mouse_pos
    cam_pos = scene.camera.pos
    ray_dir = scene.mouse.ray.norm()
    hover_threshold = 0.1  # adjust to control hover sensitivity

    # Check each point to see if it's being hovered over.
    for pt in points:
        # Calculate the distance from the point to the mouse ray.
        v = pt.pos - cam_pos
        proj_length = dot(v, ray_dir)
        closest_point = cam_pos + proj_length * ray_dir
        distance_to_ray = mag(pt.pos - closest_point)
        if distance_to_ray < hover_threshold:
            pt.radius = hover_radius
            pt.color = hover_color
        else:
            pt.radius = pt.default_radius
            pt.color = pt.default_color

    # If dragging, rotate the sphere based on mouse movement.
    if drag and prev_mouse_pos is not None:
        current_pos = scene.mouse.pos
        delta = current_pos - prev_mouse_pos
        angle = mag(delta) * rotation_sensitivity
        if angle != 0:
            # Create a rotation axis that is perpendicular to the mouse movement in screen space.
            axis = vector(-delta.y, delta.x, 0).norm()
            rotate_points(angle, axis)
        prev_mouse_pos = current_pos

# Bind mouse events
scene.bind("mousedown", on_mouse_down)
scene.bind("mouseup", on_mouse_up)
scene.bind("mousemove", on_mouse_move)

# Main loop: VPython handles the scene updates and event callbacks.
while True:
    rate(100)
