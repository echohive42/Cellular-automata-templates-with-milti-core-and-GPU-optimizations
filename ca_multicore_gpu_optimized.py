import pygame
import numpy as np
import random
from termcolor import cprint
import sys
from numba import jit, prange, cuda
import numpy.typing as npt
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import os

# Initialize Pygame and get screen dimensions
try:
    pygame.init()
    screen_info = pygame.display.Info()
    WINDOW_WIDTH = screen_info.current_w
    WINDOW_HEIGHT = screen_info.current_h
    cprint(f"Screen resolution detected: {WINDOW_WIDTH}x{WINDOW_HEIGHT}", "green")
    cprint("Pygame initialized successfully!", "green")
except pygame.error as e:
    cprint(f"Failed to initialize Pygame: {e}", "red")
    sys.exit(1)

# Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
CELL_SIZE = 2
FPS = 100

# Calculate grid dimensions
GRID_WIDTH = WINDOW_WIDTH // CELL_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

# Multiprocessing constants
NUM_CORES = max(1, mp.cpu_count() - 1)
CHUNK_SIZE = 100

# Color constants
MAX_AGE = 100
HUE_RANGE = 0.7

# Pre-calculate color lookup table with better precision
COLOR_TABLE_SIZE = 512  # Increased for better color precision

@jit(nopython=True)
def hsv_to_rgb_numba(h, s, v):
    """Numba-compatible HSV to RGB conversion"""
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q

@jit(nopython=True)
def generate_color_lookup_numba():
    """Generate color lookup table using Numba-optimized HSV to RGB conversion"""
    lookup = np.zeros((COLOR_TABLE_SIZE, COLOR_TABLE_SIZE, 3), dtype=np.uint8)
    for age in range(COLOR_TABLE_SIZE):
        for neighbors in range(COLOR_TABLE_SIZE):
            age_norm = age / COLOR_TABLE_SIZE
            neighbor_norm = neighbors / COLOR_TABLE_SIZE
            hue = 0.7 - (0.7 * age_norm * 0.7 + neighbor_norm * 0.3)
            saturation = 0.8 + (neighbor_norm * 0.2)
            value = 0.8 + (age_norm * 0.2)
            r, g, b = hsv_to_rgb_numba(hue, saturation, value)
            lookup[age, neighbors, 0] = int(r * 255)
            lookup[age, neighbors, 1] = int(g * 255)
            lookup[age, neighbors, 2] = int(b * 255)
    return lookup

# Generate color lookup table with Numba optimization
try:
    cprint("Generating color lookup table...", "yellow")
    color_lookup = generate_color_lookup_numba()
    cprint("Color lookup table generated successfully!", "green")
except Exception as e:
    cprint(f"Error generating color lookup table: {e}", "red")
    sys.exit(1)

# Check CUDA availability
CUDA_AVAILABLE = False
try:
    if cuda.is_available():
        # Test CUDA functionality
        test_array = np.zeros((10, 10), dtype=np.int32)
        d_test = cuda.to_device(test_array)
        d_test.copy_to_host()
        CUDA_AVAILABLE = True
        cprint("CUDA support enabled and tested successfully!", "green")
except Exception as e:
    cprint(f"CUDA not available or error in CUDA initialization: {e}", "yellow")
    cprint("Falling back to CPU-only mode", "yellow")

if CUDA_AVAILABLE:
    @cuda.jit
    def count_neighbors_cuda(grid, neighbors):
        i, j = cuda.grid(2)
        rows, cols = grid.shape
        
        if i < rows and j < cols:
            count = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % rows
                    nj = (j + dj) % cols
                    count += grid[ni, nj]
            neighbors[i, j] = count

@jit(nopython=True, parallel=True)
def count_neighbors_parallel(grid):
    rows, cols = grid.shape
    neighbors = np.zeros_like(grid)
    for i in prange(rows):
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

@jit(nopython=True)
def process_chunk_optimized(chunk, neighbors_chunk, age_chunk):
    new_chunk = np.zeros_like(chunk)
    new_age = np.zeros_like(age_chunk)
    
    alive_mask = chunk == 1
    dead_mask = chunk == 0
    
    # Survival rules
    survive_mask = alive_mask & ((neighbors_chunk >= 2) & (neighbors_chunk <= 3))
    new_chunk[survive_mask] = 1
    new_age[survive_mask] = np.minimum(age_chunk[survive_mask] + 1, MAX_AGE)
    
    # Birth rules
    birth_mask = dead_mask & (neighbors_chunk == 3)
    new_chunk[birth_mask] = 1
    
    return new_chunk, new_age

def update_grid_parallel(grid, age_matrix):
    try:
        if CUDA_AVAILABLE:
            try:
                # Use CUDA for neighbor counting
                neighbors = np.zeros_like(grid)
                threadsperblock = (16, 16)
                blockspergrid_x = (grid.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
                blockspergrid_y = (grid.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                
                d_grid = cuda.to_device(grid)
                d_neighbors = cuda.to_device(neighbors)
                count_neighbors_cuda[blockspergrid, threadsperblock](d_grid, d_neighbors)
                neighbors = d_neighbors.copy_to_host()
            except Exception as e:
                cprint(f"CUDA operation failed, falling back to CPU: {e}", "yellow")
                neighbors = count_neighbors_parallel(grid)
        else:
            neighbors = count_neighbors_parallel(grid)
        
        # Process entire grid at once using vectorized operations
        new_grid = np.zeros_like(grid)
        new_age_matrix = np.zeros_like(age_matrix)
        
        # Vectorized rules application
        alive_mask = grid == 1
        dead_mask = ~alive_mask
        
        # Survival rules
        survive_mask = alive_mask & ((neighbors >= 2) & (neighbors <= 3))
        new_grid[survive_mask] = 1
        new_age_matrix[survive_mask] = np.minimum(age_matrix[survive_mask] + 1, MAX_AGE)
        
        # Birth rules
        birth_mask = dead_mask & (neighbors == 3)
        new_grid[birth_mask] = 1
        
        return new_grid, new_age_matrix
        
    except Exception as e:
        cprint(f"Error in parallel grid update: {e}", "red")
        return grid, age_matrix

@jit(nopython=True, parallel=True)
def create_color_array(grid, age_matrix, neighbors):
    rows, cols = grid.shape
    colors = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for i in prange(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                age_idx = min(int((age_matrix[i, j] / MAX_AGE) * (COLOR_TABLE_SIZE - 1)), COLOR_TABLE_SIZE - 1)
                neighbor_idx = min(int((neighbors[i, j] / 8) * (COLOR_TABLE_SIZE - 1)), COLOR_TABLE_SIZE - 1)
                colors[i, j] = color_lookup[age_idx, neighbor_idx]
    
    return colors

def create_surface_from_grid_parallel(grid, age_matrix):
    try:
        if CUDA_AVAILABLE:
            try:
                neighbors = np.zeros_like(grid)
                threadsperblock = (16, 16)
                blockspergrid_x = (grid.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
                blockspergrid_y = (grid.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                
                d_grid = cuda.to_device(grid)
                d_neighbors = cuda.to_device(neighbors)
                count_neighbors_cuda[blockspergrid, threadsperblock](d_grid, d_neighbors)
                neighbors = d_neighbors.copy_to_host()
            except Exception as e:
                cprint(f"CUDA operation failed, falling back to CPU: {e}", "yellow")
                neighbors = count_neighbors_parallel(grid)
        else:
            neighbors = count_neighbors_parallel(grid)
        
        colors = create_color_array(grid, age_matrix, neighbors)
        
        # Create surface with double buffering
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        temp_array = np.repeat(np.repeat(colors, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
        
        # Optimize surface array creation with direct buffer access
        pygame.surfarray.blit_array(surface, np.transpose(temp_array, (1, 0, 2)))
        
        return surface
    except Exception as e:
        cprint(f"Error creating surface: {e}", "red")
        return None

def initialize_grid():
    try:
        grid = np.random.choice([0, 1], GRID_WIDTH * GRID_HEIGHT, p=[0.85, 0.15]).reshape(GRID_HEIGHT, GRID_WIDTH)
        age_matrix = np.zeros_like(grid)
        cprint("Grid initialized successfully!", "cyan")
        return grid, age_matrix
    except Exception as e:
        cprint(f"Error initializing grid: {e}", "red")
        sys.exit(1)

def main():
    try:
        # Set process priority
        if sys.platform == 'win32':
            try:
                import psutil
                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                cprint("Process priority set to high", "green")
            except:
                cprint("Could not set process priority", "yellow")
        
        # Initialize display with double buffering
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME | pygame.DOUBLEBUF | pygame.HWSURFACE)
        screen = pygame.display.get_surface()
        screen.set_alpha(None)  # Disable alpha for performance
        pygame.display.set_caption(f"Cellular Automata (Using {NUM_CORES} cores{'+ CUDA' if CUDA_AVAILABLE else ''})")
        
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 36)
        
        grid, age_matrix = initialize_grid()
        
        # Pre-compile Numba functions
        cprint("Pre-compiling Numba functions...", "yellow")
        _ = count_neighbors_parallel(np.zeros((10, 10)))
        _ = create_color_array(np.zeros((10, 10)), np.zeros((10, 10)), np.zeros((10, 10)))
        cprint("Numba compilation complete!", "green")
        
        running = True
        paused = False
        frame_count = 0
        fps_update_freq = 30
        fps_values = []
        fps_history_size = 5
        last_time = pygame.time.get_ticks()
        
        # Initialize FPS display
        fps_surface = font.render("FPS: 0", True, WHITE)
        fps_background = pygame.Surface((200, 50))
        fps_background.fill(BLACK)
        fps_background.set_alpha(128)
        
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
                grid, age_matrix = update_grid_parallel(grid, age_matrix)

            surface = create_surface_from_grid_parallel(grid, age_matrix)
            if surface:
                screen.blit(surface, (0, 0))
            
            # Update FPS display
            frame_count += 1
            if frame_count >= fps_update_freq:
                current_time = pygame.time.get_ticks()
                fps = frame_count * 1000 / (current_time - last_time)
                fps_values.append(fps)
                
                if len(fps_values) > fps_history_size:
                    fps_values.pop(0)
                
                avg_fps = int(sum(fps_values) / len(fps_values))
                fps_surface = font.render(
                    f"FPS: {avg_fps} ({NUM_CORES} cores{'+ CUDA' if CUDA_AVAILABLE else ''})",
                    True, WHITE
                )
                
                frame_count = 0
                last_time = current_time
            
            # Draw FPS with pre-created background
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
    os.environ['OMP_NUM_THREADS'] = str(NUM_CORES)
    main()