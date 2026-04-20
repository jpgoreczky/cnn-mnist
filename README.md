# Assignment 3 – Convolutional Neural Network for MNIST Digit Classification

**Course:** CAP 4453 – Robot Vision  
**Language:** Python (Jupyter Notebook)  
**Libraries:** TensorFlow, Keras, NumPy  
**Topic:** Image classification using Convolutional Neural Networks (CNNs)

---

## Overview

This project implements and compares two CNN architectures to classify handwritten digits from the MNIST dataset. Both models are built using TensorFlow/Keras, trained on 60,000 images, and evaluated on 10,000 test images across 10 digit classes (0–9). Part B's deeper architecture achieved higher test accuracy.

---

## Models

### Part A – Compact CNN
A lightweight model with two convolutional blocks and direct classification via softmax.

| Layer        | Config                       |
|--------------|------------------------------|
| Conv2D       | 32 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool size                |
| Conv2D       | 64 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool size                |
| Flatten      | —                            |
| Dropout      | 50% rate                     |
| Dense        | 10 neurons, Softmax          |

---

### Part B – Deeper CNN ✅ Higher Accuracy
A deeper model with larger initial filters and two fully connected hidden layers before the output.

| Layer        | Config                       |
|--------------|------------------------------|
| Conv2D       | 30 filters, 5×5 kernel, ReLU |
| MaxPooling2D | 2×2 pool size                |
| Conv2D       | 15 filters, 3×3 kernel, ReLU |
| MaxPooling2D | 2×2 pool size                |
| Dropout      | 20% rate                     |
| Flatten      | —                            |
| Dense        | 128 neurons, ReLU            |
| Dense        | 50 neurons, ReLU             |
| Dense        | 10 neurons, Softmax          |

---

## Why Part B Performs Better

Part B outperforms Part A for three key reasons:

1. **Larger initial kernel (5×5 vs 3×3):** The first convolutional layer captures a wider spatial context from the start, allowing it to identify features more effectively early in the network.

2. **Deeper fully connected head:** Part B adds two dense layers (128 and 50 neurons) before the output, giving the model more capacity to combine and interpret the features learned by the convolutional layers. Part A jumps directly from the convolutional layers to the 10-class output.

3. **Lower dropout rate (20% vs 50%):** Part A's aggressive 50% dropout discards too much information during training. Part B's 20% rate still prevents overfitting while retaining more learned signal each pass.

---

## Training Configuration

| Parameter        | Value                     |
|------------------|---------------------------|
| Dataset          | MNIST (28×28 grayscale)   |
| Training samples | 60,000                    |
| Test samples     | 10,000                    |
| Classes          | 10 (digits 0–9)           |
| Batch size       | 128                       |
| Epochs           | 2                         |
| Optimizer        | Adam                      |
| Loss function    | Categorical cross-entropy |
| Validation split | 10%                       |

---

## How to Run

**Install dependencies:**
```bash
pip install tensorflow numpy jupyter
```

**Run the notebook:**
```bash
jupyter notebook Robot_Vision_Assignment_3.ipynb
```

Or run the script directly:
```bash
python robot_vision_assignment_3.py
```

---

## Project Structure

```
assignment3_cnn/
├── mnist_cnn_classification.ipynb  # Full notebook: Part A, Part B, and conclusion
└── partA_result.png                 # Training result screenshot for Part A
```

---

## Key Concepts Demonstrated

- Building CNN architectures with the Keras Sequential API
- Convolutional layers for spatial feature extraction
- MaxPooling for downsampling and translation invariance
- Dropout regularization to prevent overfitting
- Fully connected (Dense) layers for classification
- Softmax output for multi-class probability distribution
- Comparative architecture analysis: depth, kernel size, and regularization tradeoffs