import pygame
import numpy as np
import random
from termcolor import cprint
import sys
from colorsys import hsv_to_rgb
from numba import jit
import numpy.typing as npt

# Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
CELL_SIZE = 2
FPS = 100

# Color constants
MAX_AGE = 100  # Maximum age to track
HUE_RANGE = 0.7  # Range of hues to use (0.7 gives a nice blue-green-yellow-red range)

# Pre-calculate color lookup table for better performance
COLOR_TABLE_SIZE = 256
color_lookup = np.zeros((COLOR_TABLE_SIZE, COLOR_TABLE_SIZE, 3), dtype=np.uint8)
for age in range(COLOR_TABLE_SIZE):
    for neighbors in range(COLOR_TABLE_SIZE):
        age_norm = age / COLOR_TABLE_SIZE
        neighbor_norm = neighbors / COLOR_TABLE_SIZE
        hue = 0.7 - (0.7 * age_norm * 0.7 + neighbor_norm * 0.3)
        saturation = 0.8 + (neighbor_norm * 0.2)
        value = 0.8 + (age_norm * 0.2)
        rgb = hsv_to_rgb(hue, saturation, value)
        color_lookup[age, neighbors] = (np.array(rgb) * 255).astype(np.uint8)

try:
    # Initialize Pygame
    pygame.init()
    # Get the screen info
    screen_info = pygame.display.Info()
    WINDOW_WIDTH = screen_info.current_w
    WINDOW_HEIGHT = screen_info.current_h
    # Calculate grid size based on screen dimensions and cell size
    GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
    GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE
    cprint(f"Screen resolution detected: {WINDOW_WIDTH}x{WINDOW_HEIGHT}", "green")
    cprint("Pygame initialized successfully!", "green")
except pygame.error as e:
    cprint(f"Failed to initialize Pygame: {e}", "red")
    sys.exit(1)

@jit(nopython=True)
def count_neighbors(grid):
    """Optimized neighbor counting with Numba"""
    rows, cols = grid.shape
    neighbors = np.zeros_like(grid)
    for i in range(rows):
        for j in range(cols):
            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % rows
                    nj = (j + dj) % cols
                    count += grid[ni, nj]
            neighbors[i, j] = count
    return neighbors

def initialize_grid():
    """Initialize random grid and age matrix"""
    try:
        # Initialize main grid and age matrix
        grid = np.random.choice([0, 1], GRID_WIDTH * GRID_HEIGHT, p=[0.85, 0.15]).reshape(GRID_HEIGHT, GRID_WIDTH)
        age_matrix = np.zeros_like(grid)
        cprint("Grid initialized successfully!", "cyan")
        return grid, age_matrix
    except Exception as e:
        cprint(f"Error initializing grid: {e}", "red")
        sys.exit(1)

def update_grid(grid, age_matrix):
    """Update grid and age matrix based on Conway's Game of Life rules"""
    try:
        # Count neighbors using optimized function
        neighbors = count_neighbors(grid)

        # Create new grid based on rules (vectorized operations)
        new_grid = np.zeros_like(grid)
        new_grid = np.where((grid == 1) & ((neighbors < 2) | (neighbors > 3)), 0, grid)
        new_grid = np.where((grid == 0) & (neighbors == 3), 1, new_grid)

        # Update age matrix (vectorized operations)
        new_age_matrix = np.where(new_grid == 1, age_matrix + 1, 0)
        new_age_matrix = np.minimum(new_age_matrix, MAX_AGE)

        return new_grid, new_age_matrix
    except Exception as e:
        cprint(f"Error updating grid: {e}", "red")
        return grid, age_matrix

def create_surface_from_grid(grid, age_matrix):
    """Create a surface from the grid using color coding based on age and structure size"""
    try:
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Count neighbors for color mapping (using optimized function)
        neighbors = count_neighbors(grid)
        
        # Create color array using pre-calculated lookup table
        colors = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.uint8)
        live_mask = grid == 1
        
        if np.any(live_mask):
            # Scale indices to lookup table size
            age_indices = ((age_matrix[live_mask] / MAX_AGE) * (COLOR_TABLE_SIZE - 1)).astype(np.uint8)
            neighbor_indices = ((neighbors[live_mask] / 24) * (COLOR_TABLE_SIZE - 1)).astype(np.uint8)
            
            # Use lookup table for colors
            colors[live_mask] = color_lookup[age_indices, neighbor_indices]
        
        # Create the surface array with correct dimensions (optimized)
        temp_array = np.repeat(np.repeat(colors, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
        
        # Optimize surface array creation
        surf_array = pygame.surfarray.pixels3d(surface)
        surf_array[:] = np.transpose(temp_array, (1, 0, 2))
        del surf_array
        
        return surface
    except Exception as e:
        cprint(f"Error creating surface: {e}", "red")
        return None

def main():
    try:
        # Set up display in borderless windowed mode
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME)
        pygame.display.set_caption("Cellular Automata")
        clock = pygame.time.Clock()
        
        # Initialize font for FPS display
        try:
            font = pygame.font.Font(None, 36)
            cprint("Font initialized successfully!", "green")
        except Exception as e:
            cprint(f"Error initializing font: {e}", "red")
            font = None
        
        # Initialize grid and age matrix
        grid, age_matrix = initialize_grid()
        
        # Pre-compile Numba functions
        cprint("Pre-compiling Numba functions...", "yellow")
        _ = count_neighbors(np.zeros((10, 10)))
        cprint("Numba compilation complete!", "green")
        
        running = True
        paused = False
        
        # FPS tracking variables
        frame_count = 0
        fps_update_freq = 30  # Update FPS display every 30 frames
        fps_values = []  # Store last 5 FPS values for averaging
        fps_history_size = 5
        last_time = pygame.time.get_ticks()
        current_fps_display = "FPS: 0"  # Store current FPS text
        fps_surface = None  # Store the rendered FPS surface
        fps_background = None  # Store the background rectangle
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    grid_x, grid_y = y // CELL_SIZE, x // CELL_SIZE
                    if grid_x < GRID_HEIGHT and grid_y < GRID_WIDTH:
                        grid[grid_x][grid_y] = 1 - grid[grid_x][grid_y]
                        age_matrix[grid_x][grid_y] = 0

            if not paused:
                grid, age_matrix = update_grid(grid, age_matrix)

            # Create and draw surface
            surface = create_surface_from_grid(grid, age_matrix)
            if surface:
                screen.blit(surface, (0, 0))
            
            # Update FPS calculation
            frame_count += 1
            if frame_count >= fps_update_freq:
                current_time = pygame.time.get_ticks()
                fps = frame_count * 1000 / (current_time - last_time)
                fps_values.append(fps)
                
                # Keep only last few FPS values
                if len(fps_values) > fps_history_size:
                    fps_values.pop(0)
                
                # Calculate average FPS
                avg_fps = int(sum(fps_values) / len(fps_values))
                
                # Only update display if FPS changed significantly (more than 2 FPS)
                new_fps_text = f"FPS: {avg_fps}"
                if new_fps_text != current_fps_display:
                    current_fps_display = new_fps_text
                    if font:
                        fps_surface = font.render(current_fps_display, True, (255, 255, 255))
                        fps_rect = fps_surface.get_rect(topleft=(10, 10))
                        fps_background = pygame.Surface(fps_rect.inflate(20, 20).size)
                        fps_background.fill((0, 0, 0))
                
                frame_count = 0
                last_time = current_time
            
            # Draw FPS
            if fps_surface and fps_background:
                screen.blit(fps_background, (0, 0))
                screen.blit(fps_surface, (10, 10))
            
            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()
        cprint("Game closed successfully!", "green")

    except Exception as e:
        cprint(f"An error occurred in main loop: {e}", "red")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main() 