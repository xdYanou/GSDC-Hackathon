import pygame
import random
import math
from components.obstacle import Obstacle

class CrystalObstacle(Obstacle):
    def __init__(self, pos, size):
        super().__init__(pos, size)

        # Crystal effect colors
        self.core_color = (0, 255, 255)  # Cyan
        self.glow_color = (0, 150, 255)  # Light blue
        self.accent_color = (255, 255, 255)  # White

        # Random crystal points
        self.crystal_points = self._generate_crystal_points()

        # Animation properties
        self.pulse_time = random.random() * math.pi * 2  # Random start phase
        self.pulse_speed = random.uniform(0.02, 0.05)  # Random pulse speed
        self.glow_intensity = 0

        # Particle system
        self.particles = self._init_particles()

    def _init_particles(self):
        return [(
            pygame.Vector2(
                self.pos.x + random.uniform(0, self.size[0]),  # Random position within crystal
                self.pos.y + random.uniform(0, self.size[1])
            ),
            random.uniform(1, 4)  # Random size
        ) for _ in range(random.randint(5, 15))]  # Random number of particles

    def _generate_crystal_points(self):
        center_x = self.pos.x + self.size[0] / 2
        center_y = self.pos.y + self.size[1] / 2

        points = []
        num_points = random.randint(4, 10)

        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            angle += random.uniform(-0.2, 0.2)  # Randomize angle for irregularity
            radius = min(self.size[0], self.size[1]) / 2
            radius *= random.uniform(0.5, 1.5)

            x = center_x + math.cos(angle) * radius
            y = center_y + math.sin(angle) * radius
            points.append((x, y))

        return points

    def draw(self, screen):
        # Pulse animation
        self.pulse_time += self.pulse_speed
        self.glow_intensity = (math.sin(self.pulse_time) + 1) / 2  # Value between 0 and 1

        # Outer glow
        glow_surface = pygame.Surface(self.size, pygame.SRCALPHA)
        glow_radius = min(self.size[0], self.size[1]) / 1.6
        center = (self.size[0] / 2, self.size[1] / 2)

        # Pulsing glow effect
        glow_alpha = int(120 * self.glow_intensity)
        for radius in range(int(glow_radius), 0, -1):
            alpha = int(glow_alpha * (radius / glow_radius))
            pygame.draw.circle(glow_surface, (*self.glow_color, alpha),
                               center, radius)

        screen.blit(glow_surface, self.pos)

        # Draw crystal shape
        if len(self.crystal_points) >= 3:
            # Base crystal with transparency
            crystal_alpha = int(230 + 25 * self.glow_intensity)
            crystal_color = (*self.core_color, crystal_alpha)
            pygame.draw.polygon(screen, crystal_color, self.crystal_points)

            # Highlights
            center = (self.pos.x + self.size[0] / 2, self.pos.y + self.size[1] / 2)
            for i in range(len(self.crystal_points)):
                p1 = self.crystal_points[i]
                p2 = self.crystal_points[(i + 1) % len(self.crystal_points)]

                # Accent lines with pulsing
                accent_alpha = int(180 * self.glow_intensity)
                pygame.draw.line(screen, (*self.accent_color, accent_alpha), p1, p2, 2)
                pygame.draw.line(screen, (*self.accent_color, accent_alpha), center, p1, 1)

        # Update and draw particles
        for i, (particle_pos, particle_size) in enumerate(self.particles):
            # Circular particle movement
            angle = self.pulse_time + (i * math.pi / len(self.particles))
            particle_pos.x += math.cos(angle) * 0.5
            particle_pos.y += math.sin(angle) * 0.5

            # Keep particles within bounds
            particle_pos.x = max(self.pos.x, min(self.pos.x + self.size[0], particle_pos.x))
            particle_pos.y = max(self.pos.y, min(self.pos.y + self.size[1], particle_pos.y))

            # Draw particle
            particle_alpha = int(255 * self.glow_intensity)
            pygame.draw.circle(screen, (*self.accent_color, particle_alpha),
                               particle_pos, particle_size)