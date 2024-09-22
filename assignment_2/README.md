MNIST Hogwild Project
This project implements the MNIST Hogwild training technique using PyTorch, as outlined in the PyTorch MNIST Hogwild example. The project utilizes Docker and Docker Compose to create a reproducible environment for training, evaluating, and inferring a convolutional neural network on the MNIST dataset.

Requirements
  Docker
  Docker Compose
  Python 3.x
  PyTorch




Instructions
Step 1: Build the Docker Images
Run the following command to build all the Docker images:

    docker compose build


Step 2: Run the Services
   
   Run each Docker Compose service in sequence:

    docker compose run train
    docker compose run evaluate
    docker compose run infer

Step 3: Verify the Outputs
  Checkpoint Verification:

  Check if the checkpoint file (mnist_cnn.pt) is saved in the mnist volume. If found, it will display:

  Evaluation Results:

    Check if the evaluation results file (eval_results.json) is saved in the mnist volume. An example of the contents:

    {
      "Test loss": 0.0890245330810547,
      "Accuracy": 97.12
    }

    
