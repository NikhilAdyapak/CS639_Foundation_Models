""" 
    Name: Nikhil Adyapak
    ID: 9088291571 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import struct
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

'''
CS639 - Intro to Foundation Models
Homework 1: CS 639 Assignment 1: Neural Network Fundamentals

I have included parts of HW instructions (639_HW1.pdf) as comments in the code.
Downloaded MNIST dataset from GitHub and renamed the files 
t10k-images-idx3-ubyte (different from loader code in 639_HW1.pdf; loader has t10k-images.idx3-ubyte)
loader code in hw instruction has "." instead of "-"
'''

'''
Implement a feed-forward neural network from scratch. Your model should have one hidden
layer and be capable of being initialized with any number of input, hidden and output
units/vertices.
'''

np.random.seed(0) # Prior to training, set np.random.seed(0)

# Use ReLU activation in the hidden layer
def relu(x):
    return np.maximum(0, x)

# back prop
def relu_derivative(x):
    return (x > 0).astype(float)

# When implementing softmax, ensure numerical stability by subtracting the maximum logit before exponentiation.
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # initialize the model weights from a uniform random distribution U(-0.01, 0.01).
        # The input and hidden layers should always both include a bias unit (how you implement the bias is your choice).
        self.W1 = np.random.uniform(-0.01, 0.01, (input_size + 1, hidden_size)) 
        self.W2 = np.random.uniform(-0.01, 0.01, (hidden_size + 1, output_size))

    def forward(self, X):
        # The input and hidden layers should always both include a bias unit
        batch_size = X.shape[0]
        X_bias = np.c_[X, np.ones(batch_size)]
        
        # Hidden Layer
        self.z1 = np.dot(X_bias, self.W1)

        # Use ReLU activation in the hidden layer
        self.a1 = relu(self.z1)

        # Bias for hidden layer
        self.a1_bias = np.c_[self.a1, np.ones(batch_size)]
        
        # and use a linear output layer (no activation function).
        self.z2 = np.dot(self.a1_bias, self.W2)
        return self.z2


    def backward(self, X, y, output, learning_rate, task_type = 'classification'):
        batch_size = X.shape[0]
        X_bias = np.c_[X, np.ones(batch_size)]

        # Calculate Output Error
        if task_type == 'classification':
            # For Cross-Entropy + Softmax
            probs = softmax(output)
            delta2 = (probs - y) / batch_size
        else:
            # Use linear output with mean squared error (MSE) loss
            delta2 = (output - y) / batch_size

        # Gradients for W2
        dW2 = np.dot(self.a1_bias.T, delta2)

        # Backpropagate to Hidden Layer
        delta1 = np.dot(delta2, self.W2[:-1, :].T) * relu_derivative(self.z1)

        # Gradients for W1
        dW1 = np.dot(X_bias.T, delta1)

        # Do not use regularization, momentum, or adaptive optimizers: use plain mini-batch SGD only.
        self.W1 -= learning_rate * dW1
        self.W2 -= learning_rate * dW2

    '''
    For the training algorithm, use mini-batch SGD with batch size of 32, and re-shuffle training data
    at the start of each epoch.
    '''
    def train(self, X_train, y_train, epochs, learning_rate, batch_size = 32, task_type = 'classification'):
        loss_history = []
        
        print(f"Training with LR = {learning_rate}...")

        for epoch in range(epochs):
            # re-shuffle training data at the start of each epoch.
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0

            # use mini-batch SGD with batch size of 32
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward
                output = self.forward(X_batch)
                
                # Loss & Accuracy Calculation
                if task_type == 'classification':
                    probs = softmax(output)
                    probs = np.clip(probs, 1e-15, 1 - 1e-15)
                    loss = -np.sum(y_batch * np.log(probs)) / y_batch.shape[0]
                else:
                    loss = np.mean((output - y_batch) ** 2)
                
                epoch_loss += loss
                num_batches += 1
                
                # Backward
                self.backward(X_batch, y_batch, output, learning_rate, task_type)
            
            # Epoch Summary
            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            
        return loss_history

# Download the Iris dataset
def load_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None)
    
    # representing labels as three-element one-hot vectors.
    target = pd.get_dummies(df[4]).values
    data = df.iloc[:, 0:4].values
    
    return data, target

# California housing dataset
def load_housing():
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df = pd.read_csv(url)
    
    df = df.dropna()
    if 'ocean_proximity' in df.columns:
        df = df.drop('ocean_proximity', axis=1)

    # Use linear output ... and "median_house_value" as the target variable.
    target = df['median_house_value'].values.reshape(-1, 1)
    data = df.drop('median_house_value', axis=1).values
    
    return data, target

# Standardize input features to zero mean and variance of 1
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_test_std

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

# Code for reading in MNIST
def load_mnist_data():
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            images = data.reshape(num_images, rows, cols)
        return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    try:
        # Downloaded files from GitHub are train-images.idx3-ubyte (Changes the filenames)
        X_train = load_images("train-images-idx3-ubyte") 
        y_train = load_labels("train-labels-idx1-ubyte")
        X_test = load_images("t10k-images-idx3-ubyte")
        y_test = load_labels("t10k-labels-idx1-ubyte")

        # Flatten
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Normalize to [0,1]
        X_train = X_train.astype(float) / 255.0
        X_test = X_test.astype(float) / 255.0
        
        # represent labels as 10-element one-hot vectors.
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print("MNIST files not found. Skipping MNIST.")
        return None, None, None, None
    

def run_experiment(dataset_name, X, y, task_type):
    print(f" Running Experiment: {dataset_name} ")
    
    # Randomly place 80% of the datapoints into a training dataset and the remaining 20% into a test dataset
    # Set np.random.seed(0) prior to performing the random split.
    np.random.seed(0) 
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(0.8 * X.shape[0])
    
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    # Standardize input features to zero mean and variance of 1
    # Note: MNIST is already normalized (0-1), so we skip it here.
    if dataset_name in ["Iris", "Housing"]:
        X_train, X_test = standardize(X_train_raw, X_test_raw)
        print(f"DEBUG: {dataset_name} Inputs Standardized.")
    else:
        X_train, X_test = X_train_raw, X_test_raw

    # Note: Standardizing Targets for Housing to prevent Exploding Gradients (Allowed by forum)
    if dataset_name == "Housing":
        y_train, y_test = standardize(y_train_raw, y_test_raw)
        print(f"DEBUG: {dataset_name} Targets Standardized.")
    else:
        y_train, y_test = y_train_raw, y_test_raw

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # Part A/B/C: Vary Learning Rate
    # Include four lines on your plot, one for each learning rate of 1, 1e-2, 1e-3 and 1e-8.
    lrs = [1, 1e-2, 1e-3, 1e-8] 
    plt.figure()
    
    # Track best LR for Part D
    best_lr = 1e-2 
    best_metric = -float('inf') if task_type == 'classification' else float('inf')

    print("\n>>> Varying Learning Rates:")
    for lr in lrs:
        np.random.seed(0) 
        # Using a network with 5 hidden units
        nn = NeuralNetwork(input_dim, 5, output_dim) 
        
        # create a plot of average loss ... per-epoch, over 10 training epochs.
        losses = nn.train(X_train, y_train, epochs=10, learning_rate=lr, task_type=task_type)
        plt.plot(losses, label=f'LR={lr}')
        
        # Calculate Final Metrics
        # 1. Training Set Performance
        train_output = nn.forward(X_train)
        if task_type == 'classification':
             train_probs = softmax(train_output)
             train_probs = np.clip(train_probs, 1e-15, 1 - 1e-15)
             
             train_preds = np.argmax(train_output, axis=1)
             train_actual = np.argmax(y_train, axis=1)
             train_acc = np.mean(train_preds == train_actual)
        else:
             train_loss = np.mean((train_output - y_train) ** 2)
             train_acc = 0 

        # Give the average loss across all test set datapoints for each of these four models
        test_output = nn.forward(X_test)
        if task_type == 'classification':
             test_probs = softmax(test_output)
             test_probs = np.clip(test_probs, 1e-15, 1 - 1e-15)
             test_loss = -np.sum(y_test * np.log(test_probs)) / y_test.shape[0]
             
             test_preds = np.argmax(test_output, axis=1)
             test_actual = np.argmax(y_test, axis=1)
             test_acc = np.mean(test_preds == test_actual)
             
             print(f"LR {lr} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
             
             # Track best LR (highest accuracy)
             if test_acc > best_metric:
                 best_metric = test_acc
                 best_lr = lr
        else:
             test_loss = np.mean((test_output - y_test) ** 2)
             print(f"LR {lr} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")
             
             # Track best LR (lowest MSE)
             if test_loss < best_metric:
                 best_metric = test_loss
                 best_lr = lr
        
    # Make sure your plot includes a title, x/y axis labels, and a legend.
    plt.title(f'{dataset_name}: Loss vs Epoch (Varying LR)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dataset_name}_LR_plot.png')
    plt.close()

    # Part D/E: Vary Hidden Units
    # Include four lines on your plot, one for each hidden layer size of 2, 8, 16 and 32 units.
    hidden_units = [2, 8, 16, 32] 
    
    # Using an LR of 1e-2... (Updated: Using BEST LR from Part B as permitted by post on piazza)
    # If best_lr is very small (like 1e-8), default back to 0.1 or 0.01 to ensure visible learning if needed
    if best_lr < 1e-4: best_lr = 0.01 
    
    plt.figure()
    
    print(f"\n>>> Varying Hidden Units (Using Best LR={best_lr}):")
    for h in hidden_units:
        np.random.seed(0)
        nn = NeuralNetwork(input_dim, h, output_dim)
        
        # create a plot of average loss by training epoch over 10 epochs.
        losses = nn.train(X_train, y_train, epochs=10, learning_rate=best_lr, task_type=task_type)
        plt.plot(losses, label=f'Hidden={h}')
        
        #  Calculate Final Metrics 
        # 1. Training Set Performance
        train_output = nn.forward(X_train)
        if task_type == 'classification':
             train_preds = np.argmax(train_output, axis=1)
             train_actual = np.argmax(y_train, axis=1)
             train_acc = np.mean(train_preds == train_actual)
        else:
             train_loss = np.mean((train_output - y_train) ** 2)

        # Give the average test loss for these four models, as well as the models' accuracy
        output = nn.forward(X_test)
        if task_type == 'classification':
            probs = softmax(output)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            test_loss = -np.sum(y_test * np.log(probs)) / y_test.shape[0]
            
            preds = np.argmax(output, axis=1)
            actual = np.argmax(y_test, axis=1)
            acc = np.mean(preds == actual)
            print(f"Hidden {h} | Train Acc: {train_acc:.4f} | Test Acc: {acc:.4f} | Test Loss: {test_loss:.4f}")
        else:
            # In part e, provide MSE instead of accuracy.
            test_loss = np.mean((output - y_test) ** 2)
            print(f"Hidden {h} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

    # Make sure your plot includes a title, x/y axis labels, and a legend.
    plt.title(f'{dataset_name}: Loss vs Epoch (Varying Hidden Units)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{dataset_name}_Hidden_plot.png')
    plt.close()


if __name__ == "__main__":
    # 1. Iris dataset
    try:
        X_iris, y_iris = load_iris()
        run_experiment("Iris", X_iris, y_iris, 'classification')
    except Exception as e:
        print(f"Iris Error: {e}")

    # 2. California Housing Dataset
    try:
        X_house, y_house = load_housing()
        run_experiment("Housing", X_house, y_house, 'regression')
    except Exception as e:
        print(f"Housing Error: {e}")

    # 3. MNIST dataset
    try:
        X_train_m, y_train_m, X_test_m, y_test_m = load_mnist_data()
        if X_train_m is not None:
            # Note: we will use the official 60,000 / 10,000 train/test split. Do not create a custom 80/20 split for MNIST.
            print(" Running Experiment: MNIST ")
            
            # You may train on a subset of 20,000 MNIST training examples if runtime becomes an issue
            # X_train_m = X_train_m[:20000]
            # y_train_m = y_train_m[:20000]

            input_dim = X_train_m.shape[1]
            output_dim = y_train_m.shape[1]
            
            # Manual run for MNIST since it doesn't fit standard run_experiment (no 80/20 split logic needed)
            
            # Part A: LR
            lrs = [1, 1e-2, 1e-3, 1e-8]
            best_lr = 1e-2
            best_acc = 0
            
            plt.figure()
            print("\n>>> Varying Learning Rates:")
            for lr in lrs:
                np.random.seed(0)
                nn = NeuralNetwork(input_dim, 5, output_dim)
                losses = nn.train(X_train_m, y_train_m, epochs=10, learning_rate=lr, task_type='classification')
                plt.plot(losses, label=f'LR={lr}')
                
                # Test Metrics
                output = nn.forward(X_test_m)
                probs = softmax(output)
                probs = np.clip(probs, 1e-15, 1 - 1e-15) 
                test_loss = -np.sum(y_test_m * np.log(probs)) / y_test_m.shape[0]
                
                preds = np.argmax(output, axis=1)
                actual = np.argmax(y_test_m, axis=1)
                acc = np.mean(preds == actual)
                
                print(f"MNIST LR {lr} | Test Acc: {acc:.4f} | Test Loss = {test_loss:.4f}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_lr = lr
            
            plt.title('MNIST: Loss vs Epoch (Varying LR)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('MNIST_LR_plot.png')
            plt.close()

            # Part D: Hidden Units
            hidden = [2, 8, 16, 32]
            plt.figure()
            print(f"\n>>> Varying Hidden Units (Using Best LR={best_lr}):")
            for h in hidden:
                np.random.seed(0)
                nn = NeuralNetwork(input_dim, h, output_dim)
                losses = nn.train(X_train_m, y_train_m, epochs=10, learning_rate=best_lr, task_type='classification')
                plt.plot(losses, label=f'Hidden={h}')
                
                output = nn.forward(X_test_m)
                probs = softmax(output) # Needed for loss calculation
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                test_loss = -np.sum(y_test_m * np.log(probs)) / y_test_m.shape[0]

                preds = np.argmax(output, axis=1)
                actual = np.argmax(y_test_m, axis=1)
                acc = np.mean(preds == actual)
                print(f"MNIST Hidden {h} | Test Acc = {acc:.4f} | Test Loss: {test_loss:.4f}")

            plt.title('MNIST: Loss vs Epoch (Varying Hidden)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('MNIST_Hidden_plot.png')
            plt.close()

    except Exception as e:
        print(f"MNIST Error (Make sure files are in dir): {e}")