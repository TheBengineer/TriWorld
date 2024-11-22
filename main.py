import numpy as np
import pygame
from perlin_noise import PerlinNoise
from multiprocessing import Pool

world_size = 10
number_of_workers = 1
pool = Pool(number_of_workers)

r_noise = PerlinNoise(octaves=6, seed=1)
g_noise = PerlinNoise(octaves=6, seed=2)
b_noise = PerlinNoise(octaves=6, seed=3)


def generate_row(i):
    print(f"Generating row {i}")
    coordinates = [[i, j] for j in range(world_size)]
    row_values = np.array(
        [[r_noise(coord) * 128 + 128,
          g_noise(coord) * 128 + 128,
          b_noise(coord) * 128 + 128] for coord in coordinates],
        dtype=np.uint8)
    return row_values


# populate world with random colors in perlin noise
row_coordinates = range(world_size)
rows = pool.map(generate_row, row_coordinates)
world = np.array(rows)


def display_world():
    pygame.init()
    screen = pygame.display.set_mode((world_size, world_size))
    pygame.display.set_caption("World")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.surfarray.blit_array(screen, world)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    display_world()
