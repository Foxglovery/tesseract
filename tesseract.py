from vpython import *
import math
import numpy as np

# ----------------------------
# Helper functions for 4D rotation and projection

def rotate4d(vec, angle, i, j):
    """
    Rotate the 4D vector 'vec' in the plane defined by axes i and j by 'angle' radians.
    """
    new_vec = vec.copy()
    c = math.cos(angle)
    s = math.sin(angle)
    temp_i = vec[i] * c - vec[j] * s
    temp_j = vec[i] * s + vec[j] * c
    new_vec[i] = temp_i
    new_vec[j] = temp_j
    return new_vec

def project4Dto3D(vec4):
    """
    Project a 4D point into 3D space using a simple perspective projection.
    """
    d = 3.0  # distance from the observer to the 4D object along the w-axis
    factor = d / (d - vec4[3])
    return vector(vec4[0] * factor, vec4[1] * factor, vec4[2] * factor)

# ----------------------------
# Build the tesseract (4D hypercube)

# Create the 16 vertices of the tesseract: all combinations of (±1, ±1, ±1, ±1)
vertices4d = []
for x in [-1, 1]:
    for y in [-1, 1]:
        for z in [-1, 1]:
            for w in [-1, 1]:
                vertices4d.append(np.array([x, y, z, w], dtype=float))

# Create a list of edges.
# In a hypercube, two vertices are connected if they differ by exactly one coordinate.
edges = []
n = len(vertices4d)
for i in range(n):
    for j in range(i + 1, n):
        diff = np.abs(vertices4d[i] - vertices4d[j])
        if np.count_nonzero(diff) == 1:
            edges.append((i, j))

# ----------------------------
# Set up the VPython scene

scene.title = "Tesseract Controlled by Mouse Movement"
scene.background = color.black
scene.width = 800
scene.height = 600

# Create VPython curve objects for each tesseract edge.
edge_objs = []
for edge in edges:
    curve_obj = curve(color=color.white, radius=0.02)
    curve_obj.append(vector(0, 0, 0))
    curve_obj.append(vector(0, 0, 0))
    edge_objs.append((edge, curve_obj))

# Create a floor for visual reference.
floor = box(pos=vector(0, -3, 0), size=vector(10, 0.1, 10), color=color.green)

# Create the "person" as a red sphere.
person = sphere(pos=vector(0, 0, 0), radius=0.2, color=color.red, make_trail=True)

# ----------------------------
# Now we set up mouse-controlled movement.
# The tesseract and person will update only when you move the cursor.

# Sensitivity factors (adjust these as you like)
tesseract_sensitivity = 0.1  # How much the tesseract rotates per mouse delta
person_sensitivity = 0.05    # How much the person moves per mouse delta

# Store the last mouse position to compute changes.
last_mouse = scene.mouse.pos
# We'll accumulate the person's movement in a base position.
person_base = vector(0, 0, 0)

while True:
    rate(100)  # Update at 100 frames per second

    # Get the current mouse position and compute the difference (delta) since the last frame.
    current_mouse = scene.mouse.pos
    delta = current_mouse - last_mouse

    # Use the mouse movement to set rotation angles.
    angle_xw = tesseract_sensitivity * delta.x  # rotate in the x-w plane based on horizontal movement
    angle_yz = tesseract_sensitivity * delta.y  # rotate in the y-z plane based on vertical movement

    # Rotate each 4D vertex using the new angles.
    for i in range(len(vertices4d)):
        v = vertices4d[i]
        v = rotate4d(v, angle_xw, 0, 3)  # rotate in x-w
        v = rotate4d(v, angle_yz, 1, 2)  # rotate in y-z
        vertices4d[i] = v

    # Project all 4D vertices into 3D space and update the edge curves.
    projected_vertices = [project4Dto3D(v) for v in vertices4d]
    for (edge, curve_obj) in edge_objs:
        i, j = edge
        curve_obj.modify(0, pos=projected_vertices[i])
        curve_obj.modify(1, pos=projected_vertices[j])

    # Update the "person" position based on the mouse movement.
    # Here, we update only the horizontal (x) and depth (z) positions; y remains fixed.
    person_base.x += person_sensitivity * delta.x
    person_base.z += person_sensitivity * delta.y
    person.pos = vector(person_base.x, 0, person_base.z)

    # Update last_mouse for the next iteration.
    last_mouse = current_mouse
