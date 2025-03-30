import pygame

class Obstacle:
    def __init__(self, pos, size):
        self.pos = pygame.Vector2(pos)
        self.size = size
        self.rect = pygame.Rect(self.pos, self.size)

    def draw(self, screen):
        #This method will be overridden in advanced_UI.py
        pygame.draw.rect(screen, "black", (self.pos.x, self.pos.y, self.size[0], self.size[1]))