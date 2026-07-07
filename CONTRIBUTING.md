# Contributing to SAM2-TensorRT

Thanks for your interest in contributing.

## Development Setup

### Environment Variables

The build system requires these environment variables:

```bash
export TENSORRT_ROOT=/path/to/TensorRT-10.x
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-x.x.x
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

Optionally, set `OpenCV_DIR` if CMake cannot find OpenCV automatically:

```bash
export OpenCV_DIR=/path/to/opencv/build
```

### Build

```bash
cmake -B build
cmake --build build -j$(nproc)
```

For debug builds:

```bash
cmake -B build -DDEBUG_MODE=ON
cmake --build build -j$(nproc)
```

## Code Style

- C++14 standard
- Comments in Chinese or English (both acceptable)
- Follow Google C++ Style Guide where practical
- Use `#pragma once` for header guards
- Prefer `std::unique_ptr` / `std::vector` over raw pointers where possible

## Pull Request Process

1. Fork the repository and create a feature branch from `develop`
2. Make your changes, including tests if applicable
3. Ensure the project builds cleanly: `cmake --build build`
4. Update documentation if you change public APIs
5. Submit a pull request against the `develop` branch

## Reporting Issues

When reporting bugs, please include:

- GPU model and driver version (`nvidia-smi`)
- TensorRT version
- CUDA version
- Steps to reproduce
- Error logs or crash output

## License

By contributing, you agree that your contributions will be licensed under the MIT License of this project.
