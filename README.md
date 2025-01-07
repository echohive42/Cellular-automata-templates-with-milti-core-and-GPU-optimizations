# Cellular Automata Simulator

A high-performance, GPU-accelerated implementation of Conway's Game of Life with advanced visualization features. This project includes three different implementations, each optimized for different use cases.

## Features

- **Multiple Implementations**:

  - `ca_multicore_gpu_optimized.py`: GPU-accelerated version using CUDA(MIGHT START SLOW AT FIRST)
  - `ca_multicore_optimized_.py`: Multi-core CPU optimized version
  - `ca_sonnet.py`: Basic single-core version
 
## ❤️ Support & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools & platforms (up to $180 value)

- **Performance Optimizations**:

  - CUDA GPU acceleration (when available)
  - Multi-core CPU processing
  - Numba JIT compilation
  - Vectorized operations
  - Pre-calculated color lookup tables
  - Double buffering for smooth rendering
- **Visualization Features**:

  - Dynamic color coding based on cell age and neighbor count
  - Full-screen borderless window
  - Real-time FPS counter
  - Smooth scaling and rendering
- **Interactive Controls**:

  - Space: Pause/Resume simulation
  - Escape: Exit
  - Mouse Click: Toggle cells
  - Automatic window sizing to screen resolution

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for GPU acceleration)
- Required packages listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd cellular-automata
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run any of the three implementations:

- For GPU-accelerated version:

```bash
python ca_multicore_gpu_optimized.py
```

- For multi-core CPU version:

```bash
python ca_multicore_optimized_.py
```

- For basic version:

```bash
python ca_sonnet.py
```

## Performance Comparison

- **GPU Version**: Best performance on systems with CUDA-capable GPUs
- **Multi-core Version**: Optimal for multi-core CPUs without GPU
- **Basic Version**: Good for basic usage and learning purposes

## Technical Details

### Color System

- Uses HSV to RGB color mapping
- Colors represent:
  - Cell age (from blue to red)
  - Number of neighbors (affecting saturation)
  - Combined effects create a vibrant, informative visualization

### Optimization Techniques

- JIT compilation with Numba
- Parallel processing using multiple CPU cores
- CUDA acceleration for GPU computations
- Vectorized numpy operations
- Optimized surface creation and rendering
- Double buffering for smooth display

### Memory Management

- Efficient array operations
- Pre-calculated lookup tables
- Optimized surface buffer handling
- Automatic cleanup and resource management

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Your chosen license]

## Acknowledgments

- Based on Conway's Game of Life
- Built with Python, Pygame, and Numba
- GPU acceleration powered by CUDA
