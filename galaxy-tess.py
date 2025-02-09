import pygame
import math
import random
import numpy as np

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swirling Galaxy with Orbiting Tesseracts")
clock = pygame.time.Clock()

# Define the black hole's position (center of the galaxy)
CENTER = np.array([WIDTH // 2, HEIGHT // 2], dtype=float)

# Pre-generate background stars in polar coordinates (r, theta, angular speed)
NUM_STARS = 300
stars = []
for _ in range(NUM_STARS):
    # r between 50 and 400 so stars fill a disk
    r = random.uniform(50, 400)
    theta = random.uniform(0, 2 * math.pi)
    # Inner stars rotate faster than outer ones; adjust factor as needed.
    omega = 0.002 * (400 / (r + 1))
    stars.append([r, theta, omega])


class Tesseract:
    def __init__(self, position, velocity):
        # position and velocity are 2D (screen) vectors
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.rotation_angle = 0.0  # base rotation angle for 4D rotation
        self.scale = 30  # overall size of the tesseract
        self.vertices = self.generate_vertices()  # list of 16 4D vertices
        self.edges = self.generate_edges()  # list of vertex index pairs that form edges

    def generate_vertices(self):
        # A tesseract (4D hypercube) has 16 vertices: each coordinate is either -1 or 1.
        verts = []
        for x in (-1, 1):
            for y in (-1, 1):
                for z in (-1, 1):
                    for w in (-1, 1):
                        verts.append(np.array([x, y, z, w], dtype=float))
        return verts

    def generate_edges(self):
        # Connect vertices that differ in exactly one coordinate.
        edges = []
        n = len(self.vertices)
        for i in range(n):
            for j in range(i + 1, n):
                # Count how many coordinates are the same.
                diff = np.abs(self.vertices[i] - self.vertices[j])
                if np.sum(diff == 0) == 3:
                    edges.append((i, j))
        return edges

    def rotate_in_plane(self, v, i, j, theta):
        # Rotate a 4D point v (np.array) in the plane spanned by indices i and j.
        v_new = v.copy()
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        v_new[i] = v[i] * cos_t - v[j] * sin_t
        v_new[j] = v[i] * sin_t + v[j] * cos_t
        return v_new

    def rotate_4d(self, v):
        # Apply a couple of 4D rotations.
        # Rotate in the x-w plane (indices 0 and 3)
        v = self.rotate_in_plane(v, 0, 3, self.rotation_angle)
        # Rotate in the y-z plane (indices 1 and 2) at a different speed.
        v = self.rotate_in_plane(v, 1, 2, self.rotation_angle * 0.5)
        return v

    def project_to_2d(self, v4):
        """
        Project a 4D point to 2D by doing two perspective projections:
        first from 4D to 3D (using the w coordinate), then from 3D to 2D (using z).
        """
        # 4D to 3D projection:
        d4 = 2.0  # distance from the "camera" in 4D space
        # Avoid division by zero by checking the denominator.
        factor4 = d4 / (d4 - v4[3]) if (d4 - v4[3]) != 0 else 1
        v3 = np.array([v4[0] * factor4, v4[1] * factor4, v4[2] * factor4])
        
        # 3D to 2D projection:
        d3 = 2.5  # distance from the "camera" in 3D space
        factor3 = d3 / (d3 - v3[2]) if (d3 - v3[2]) != 0 else 1
        x2d = v3[0] * factor3
        y2d = v3[1] * factor3
        return np.array([x2d, y2d])

    def get_projected_vertices(self):
        # Apply rotation, scaling, and projection to each vertex.
        projected = []
        for v in self.vertices:
            v_rot = self.rotate_4d(v)
            v_scaled = v_rot * self.scale
            v2d = self.project_to_2d(v_scaled)
            projected.append(v2d)
        return projected

    def update(self, dt, center):
        # Update the tesseract's internal rotation.
        self.rotation_angle += 0.02

        # Compute gravitational acceleration toward the center.
        dx = center[0] - self.position[0]
        dy = center[1] - self.position[1]
        r = math.sqrt(dx * dx + dy * dy) + 0.1  # add small term to avoid div-by-zero

        # Inverse-square law: a = G / r^2 (here written as G * (dx,dy) / r^3).
        G = 5000  # gravitational constant; tweak for stronger/weaker pull
        ax = G * dx / (r ** 3)
        ay = G * dy / (r ** 3)

        # Update velocity using the computed acceleration.
        self.velocity[0] += ax * dt
        self.velocity[1] += ay * dt

        # If the tesseract is near the center, apply damping to simulate it slowing down.
        if r < 150:
            self.velocity *= 0.99

        # Update the position.
        self.position += self.velocity * dt

    def draw(self, surface):
        # Get projected 2D vertices and translate them to the tesseract's simulation position.
        projected = self.get_projected_vertices()
        translated = [p + self.position for p in projected]

        # Draw the edges as lines.
        for edge in self.edges:
            start = translated[edge[0]]
            end = translated[edge[1]]
            pygame.draw.line(surface, (0, 255, 0),
                             (int(start[0]), int(start[1])),
                             (int(end[0]), int(end[1])), 1)


def spawn_tesseract(mouse_pos):
    """Spawn a new tesseract at the clicked position with an initial orbital velocity."""
    pos = np.array(mouse_pos, dtype=float)
    # Compute the vector from the center (black hole) to the mouse position.
    offset = pos - CENTER
    r = np.linalg.norm(offset)
    if r != 0:
        # A perpendicular vector to (dx,dy) is (-dy, dx). Normalize it.
        perp = np.array([-offset[1], offset[0]])
        perp /= np.linalg.norm(perp)
        # For a stable orbit in an inverse-square field, the ideal orbital speed is ~sqrt(G/r).
        # (This is not physically perfect here, but gives a nice effect.)
        G = 5000
        speed = math.sqrt(G / r)
        velocity = perp * speed
    else:
        velocity = np.array([0, 0], dtype=float)
    return Tesseract(pos, velocity)


def update_and_draw_stars(surface):
    """Update positions of background stars and draw them."""
    for star in stars:
        r, theta, omega = star
        theta += omega  # update the angle
        star[1] = theta  # store the new angle
        # Convert polar coordinates to Cartesian coordinates relative to the center.
        x = CENTER[0] + r * math.cos(theta)
        y = CENTER[1] + r * math.sin(theta)
        # Draw the star (a small white dot).
        pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), 1)


def main():
    running = True
    # List to hold active tesseracts.
    tesseracts = []

    while running:
        dt = clock.tick(60) / 1000.0  # seconds elapsed since last frame

        # Event handling.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # On mouse click, spawn a new tesseract at the click location.
                tesseract = spawn_tesseract(event.pos)
                tesseracts.append(tesseract)

        # Fill the background with a dark color.
        screen.fill((10, 10, 30))

        # Draw and update background stars.
        update_and_draw_stars(screen)

        # Update and draw each tesseract.
        for t in tesseracts:
            t.update(dt, CENTER)
            t.draw(screen)

        # Draw the black hole as a glowing circle at the center.
        pygame.draw.circle(screen, (20, 20, 20), (int(CENTER[0]), int(CENTER[1])), 30)
        pygame.draw.circle(screen, (0, 0, 0), (int(CENTER[0]), int(CENTER[1])), 20)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
