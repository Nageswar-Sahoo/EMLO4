# PyTorch Docker Assignment

Welcome to the PyTorch Docker Assignment! This project will guide you through creating a Docker container for a PyTorch environment, training a model on the MNIST dataset, and managing model checkpoints.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Instructions](#instructions)
  - [1. Create Dockerfile](#1-create-dockerfile)
  - [2. Train the Model](#2-train-the-model)
  - [3. Save Model Checkpoints](#3-save-model-checkpoints)
  - [4. Resume Training](#4-resume-training)
  - [5. Build and Run Docker Container](#5-build-and-run-docker-container)
- [Testing and Submission](#testing-and-submission)
- [Project Structure](#project-structure)
- [Notes](#notes)

## Overview

In this assignment, you will:
1. Create a Dockerfile for a PyTorch (CPU version) environment.
2. Keep the Docker image size under 1GB (uncompressed).
3. Train a neural network model on the MNIST dataset inside the Docker container.
4. Save the trained model checkpoints to the host system.
5. Implement the ability to resume model training from a saved checkpoint.

## Requirements

- **Docker** installed on your machine.
- Basic knowledge of **PyTorch** and **Docker**.
- Familiarity with Python for completing the training script.

## Instructions

### 1. Create Dockerfile

You need to create a `Dockerfile` that builds a lightweight PyTorch environment for training the MNIST dataset on CPU. The image size should be kept under **1GB**.

- Base image: Python 3.8 or later
- Install `torch`, `torchvision`, and other required dependencies.
- The container should execute `train.py` for training the model.

### 2. Train the Model

In the `train.py` file, you will:
- Load the MNIST dataset using `torchvision.datasets`.
- Define a simple neural network using PyTorch.
- Implement training and validation loops.
- After each epoch, save the model checkpoint.

### 3. Save Model Checkpoints

Ensure that model checkpoints are saved to the host machine. You can achieve this by mapping a volume between the container and the host system.

Checkpoints should be saved after each epoch in the `model_checkpoints/` directory.

### 4. Resume Training

Add functionality in `train.py` to resume model training from a saved checkpoint. You can pass a `--resume` flag to the script to trigger this behavior.

### 5. Build and Run Docker Container

To build and run the Docker container:

1. **Build the Docker Image**:
    ```bash
    docker build -t pytorch-docker-assignment .
    ```

2. **Run the Docker Container**:
    ```bash
    docker run -v $(pwd)/model_checkpoints:/app/model_checkpoints pytorch-docker-assignment
    ```

## Project Structure

```
├── Dockerfile                # Docker configuration file
├── requirements.txt          # Python dependencies
├── train.py                  # Training script
├── model_checkpoints/        # Directory to store model checkpoints
└── README.md                 # This file
```

Docker Hub Link : https://hub.docker.com/layers/nageswarsahoo/assignment_1/latest/images/sha256-d2861749317e9bfbf79d16818240a6c38075f1cb575c2b16f1af6d1869b6de39?context=repo
