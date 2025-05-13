**SGD Learning Rate Experimentation on MNIST and Fashion MNIST Datasets**
**Project Overview**
This project investigates the impact of varying learning rates on training dynamics and model performance using Stochastic Gradient Descent (SGD). We compare the performance of different learning rates (0.1, 0.2, 0.384, 0.48, and 0.6) on two standard datasets: MNIST and Fashion MNIST. Our experiments show that larger learning rates can accelerate convergence but may cause instability for certain datasets. The project empirically analyzes training and test losses, accuracy trends, and gradient behaviors under different step sizes.

**Team Members**
Jagadeesh Mandala

Anish Reddy Yellakonda

Surya Sai Nagandla

**Abstract**
This study explores the impact of learning rate on the training dynamics and performance of neural networks, inspired by the paper "SGD with Large Step Sizes Learns Sparse Features" by Andriushchenko et al. While conventional practices favor small learning rates (e.g., 0.00001) for stability, our project demonstrates that larger learning rates (0.1–0.3) can still achieve efficient convergence under certain dataset conditions. Using both MNIST and Fashion MNIST datasets, we empirically show that larger step sizes lead to faster convergence and improved accuracy for MNIST, but trigger instability and exploding gradients for Fashion MNIST. We analyze training and test loss behaviors, accuracy trends, and gradient responses under varying step sizes.

The key finding is the dataset-specific behavior of learning rates—demonstrating that while 0.3 works best for MNIST, Fashion MNIST favors smaller values (0.1 or 0.2) to avoid gradient divergence. These results affirm the paper’s theoretical insights: that SGD with large step sizes induces implicit regularization via stochastic dynamics, leading to sparse feature representations. However, the benefits strongly depend on the dataset’s complexity and sensitivity to gradient updates.

**Setup Instructions**
**Clone the repository**
To get started, clone the repository using the following command:

bash
Copy
git clone https://github.com/JagadeeshMandala/ML-PROJECT-ON-LARGE-STEP-SIZES/.git
cd ML-PROJECT-ON-LARGE-STEP-SIZES
**Dependencies**
This project requires the following libraries to run:

Python 3.x

PyTorch

Matplotlib

Pandas

NumPy

Scikit-learn

You can install the required dependencies using:

bash
Copy
pip install -r requirements.txt
Run the Code
Load and preprocess datasets
The MNIST and Fashion MNIST datasets are downloaded using torchvision.datasets. The datasets are preprocessed, normalized, and split into training and test sets.

Model Architecture
A simple feedforward neural network is used for training and testing. The architecture consists of:

Input layer: 784 (flattened 28x28 image)

Hidden layer 1: 256 neurons (ReLU)

Hidden layer 2: 128 neurons (ReLU)

Output layer: 10 neurons

Training and Testing
The model is trained with different learning rates (0.1, 0.2, 0.384, 0.48, 0.6) and tested on both MNIST and Fashion MNIST datasets. Training and test loss, accuracy (MAPE), and the behavior of the gradients are recorded.

Run script
You can run the training script with:

bash
Copy
python train.py
Reproducibility
To ensure the reproducibility of results:

Set the random seed for PyTorch and NumPy for consistent results across runs.

Ensure that the same datasets are used (MNIST and Fashion MNIST).

Environment Setup
Python version: 3.x

PyTorch: 1.x

CUDA version: 11.x (if applicable)

bash
Copy
# Optional: Set seed for reproducibility
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
Folder Structure
Experiments and Results
Learning Rates Tested:
0.1

0.2

0.384

0.48

0.6

Key Findings:
Larger learning rates (0.2 and 0.3) improve convergence and performance for MNIST.

Higher learning rates (0.48 and 0.6) cause instability and poor performance on Fashion MNIST due to exploding gradients.

Gradient clipping and learning rate scheduling improved the robustness of the training.

Example Results:
**Learning Rate    	Test Loss	  Accuracy (MAPE)**
0.1	               0.1414	        97.63%
0.2	               0.1443	        97.49%
0.384	             0.1643	        96.92%

Conclusion
This study shows the significant effect of learning rate on training stability and convergence speed in neural networks. Larger learning rates may accelerate convergence but can cause instability, especially on more complex datasets. The dataset's complexity and sensitivity to gradient updates should guide the choice of learning rate.
