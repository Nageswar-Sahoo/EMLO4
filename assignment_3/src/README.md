# Cat-Dog Classifier Project

This project is a PyTorch Lightning-based image classification model for identifying dog breeds using the Kaggle dataset. It utilizes Docker to provide an isolated environment for training, evaluation, and inference.

## Features

- **Training**: Train the Cat-Dog Classifier model using the Kaggle dataset.
- **Evaluation**: Load a saved model checkpoint and evaluate it on the validation dataset.
- **Inference**: Run inference on 10 sample images to predict their breeds.
- **Docker**: The entire project runs in a Docker container for easy setup and reproducibility.

## Dataset

We are using the **Dog Breed Image Dataset** from Kaggle: [Kaggle Dataset Link](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset).

Docker Setup
The project is dockerized, so all dependencies are encapsulated within the Docker container. Follow the steps below to set up and run the project.

    Build the Docker Image 
       docker build -t catdog-classifier .
    Train the Model
       docker run -v $(pwd)/dataset:/workspace/dataset -v $(pwd)/logs:/workspace/logs catdog-classifier



