# Dog Breed Classifier

This project implements a dog breed classifier using PyTorch Lightning. It uses `uv` for dependency management with `pyproject.toml`.

## Project Setup

This project uses `pyproject.toml` for dependency management. Make sure you have a `pyproject.toml` file in your project root with your dependencies listed. For example:

```toml
[project]
name = "dog-breed-classifier"
version = "0.1.0"
description = "A dog breed classifier using PyTorch Lightning"
dependencies = [
    "torch",
    "torchvision",
    "pytorch-lightning",
    "pillow",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Using Docker with uv and Volume Mounts

We use Docker with `uv` for dependency management and volume mounts to ensure data persistence and easy access to your model outputs.

### Build the Docker image

Use the following command to build the Docker image, specifying the correct Dockerfile name:

```bash
docker build -t dog-breed-classifier -f Dockerfile.train .
```

### Prepare Directories for Volume Mounts

Before running the Docker container, make sure you have the necessary directories on your host machine:

```bash
mkdir -p data checkpoints logs
```

### Train the model

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  dog-breed-classifier python train.py
```

### Evaluate the model

```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  dog-breed-classifier ./run_eval.sh
```

### Run inference on 10 images

```bash
docker run -v $(pwd)/data:/app/data/input \
    -v $(pwd)/data:/app/data/output \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    dogbreed-classifier python src/infer.py
```

## Volume Mounts Explanation

In each `docker run` command, we use three volume mounts:

1. `-v $(pwd)/data:/app/data`: Mounts your local `data` directory to `/app/data` in the container.
2. `-v $(pwd)/checkpoints:/app/checkpoints`: Mounts your local `checkpoints` directory to `/app/checkpoints` in the container.
3. `-v $(pwd)/logs:/app/logs`: Mounts your local `logs` directory to `/app/logs` in the container.

These mounts ensure that:
- Your data is accessible to the container
- Model checkpoints are saved on your local machine
- Logs are preserved between runs

## Preparing Your Data

1. Place your training and validation datasets in the `data` directory.
2. For inference, place your test images in `data/test_images`.

## Accessing Results

After running the Docker commands:
- Trained model checkpoints will be in the `checkpoints` directory
- Logs will be in the `logs` directory
- Inference results will be printed to the console

Remember to use the `--gpus all` flag if you want to use GPU acceleration and have the necessary NVIDIA drivers and Docker GPU support installed.