import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas
import os
from tkinter import Button
import pickle

# Paths to your image folders
banana_folder = 'banana'
apple_folder = 'apple'
input_image_path = 'input.png'

def load_images(folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize(img_size)  # Resize to img_size
            img_array = np.array(img).flatten() / 255.0  # Flatten and normalize
            images.append(img_array)
            labels.append(label)
    return images, labels

# Load images and prepare the dataset
banana_images, banana_labels = load_images(banana_folder, 0)  # Label 0 for banana
apple_images, apple_labels = load_images(apple_folder, 1)     # Label 1 for apple

X = np.array(banana_images + apple_images)
y = np.array(banana_labels + apple_labels)

# Neural Network class definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_hidden = np.random.rand(1, hidden_size) - 0.5
        self.bias_output = np.random.rand(1, output_size) - 0.5
        self.learning_rate = learning_rate

        # Load weights if they exist
        self.load_weights()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Hidden layer
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        # Output layer
        output = self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)
        return output

    def backpropagation(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        self.weights_hidden_output += self.hidden.T.dot(output_delta) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backpropagation(X, y, output)

            if epoch % 10 == 0:
                # Calculate mean squared error
                mse = np.mean(np.square(y - output))
                print(f"Epoch {epoch+1}/{epochs}, Error: {mse:.4f}")

        self.save_weights()

    def predict(self, X):
        return self.feedforward(X)
    
    def save_weights(self):
        # Save weights and biases to a file
        weights = {
            "weights_input_hidden": self.weights_input_hidden,
            "bias_hidden": self.bias_hidden,
            "weights_hidden_output": self.weights_hidden_output,
            "bias_output": self.bias_output
        }
        with open("weights.pkl", "wb") as f:
            pickle.dump(weights, f)
        print("Weights saved to 'weights.pkl'.")

    def load_weights(self):
        # Load weights and biases if the file exists
        if os.path.exists("weights.pkl"):
            with open("weights.pkl", "rb") as f:
                weights = pickle.load(f)
                self.weights_input_hidden = weights["weights_input_hidden"]
                self.bias_hidden = weights["bias_hidden"]
                self.weights_hidden_output = weights["weights_hidden_output"]
                self.bias_output = weights["bias_output"]
            print("Weights loaded from 'weights.pkl'.")
        else:
            print("No saved weights found. Initializing with random weights.")

# Prepare the labels as one-hot encoded arrays
y_onehot = np.zeros((y.size, 2))
y_onehot[np.arange(y.size), y] = 1

# Initialize and train the neural network
input_size = 64 * 64  # Assuming 28x28 images
hidden_size = 764
output_size = 2  # Two classes: apple or banana

nn = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
#nn.train(X, y_onehot, epochs=5000)

# Load and preprocess the input image
input_img = Image.open(input_image_path).convert('L')  # Convert to grayscale
input_img = input_img.resize((64, 64))
input_img_array = np.array(input_img).flatten() / 255.0

def predict_image():
    _ = 0

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 400, 400), fill="white")
    confidence_apple.set(0)
    confidence_banana.set(0)

# Initialize tkinter window
root = tk.Tk()
root.title("Apple vs Banana Classifier")

# Drawing area
canvas = Canvas(root, width=400, height=400, bg="white")
canvas.grid(row=0, column=0, rowspan=4, padx=10, pady=10)

# Image and drawing handler for PIL
img = Image.new("RGB", (400, 400), "white")
draw = ImageDraw.Draw(img)

# Draw on canvas
def paint(event):
    x, y = event.x, event.y
    r = 15  # radius of drawing point
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", width=0)
    draw.ellipse((x - r, y - r, x + r, y + r), fill="black")

    # Resize to 64x64, flatten, and normalize
    img_resized = img.resize((64, 64)).convert('L')
    img_array = np.array(img_resized).flatten() / 255.0
    prediction = nn.predict([img_array])
    confidence_apple.set(100 * (prediction[0][1]) / (prediction[0][1] + prediction[0][0]))  # Confidence for "apple" class
    confidence_banana.set(100 * (prediction[0][0]) / (prediction[0][1] + prediction[0][0]))  # Confidence for "banana" class

# Function to save the drawing as an example for training
def save_image(label):
    img_resized = img.resize((64, 64)).convert('L')
    if label == "banana":
        img_path = os.path.join(banana_folder, f"banana_{len(os.listdir(banana_folder))}.png")
    elif label == "apple":
        img_path = os.path.join(apple_folder, f"apple_{len(os.listdir(apple_folder))}.png")
    img_resized.save(img_path)

    banana_images, banana_labels = load_images(banana_folder, 0)  # Label 0 for banana
    apple_images, apple_labels = load_images(apple_folder, 1)     # Label 1 for apple

    X = np.array(banana_images + apple_images)
    y = np.array(banana_labels + apple_labels)
    # Prepare the labels as one-hot encoded arrays
    y_onehot = np.zeros((y.size, 2))
    y_onehot[np.arange(y.size), y] = 1

    nn.train(X, y_onehot, epochs=300)

canvas.bind("<B1-Motion>", paint)

# Prediction and confidence bars
confidence_apple = tk.DoubleVar()
confidence_banana = tk.DoubleVar()

ttk.Label(root, text="Apple Confidence").grid(row=0, column=1)
apple_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=confidence_apple)
apple_bar.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="Banana Confidence").grid(row=2, column=1)
banana_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", variable=confidence_banana)
banana_bar.grid(row=3, column=1, padx=10, pady=5)

# Prediction button
predict_button = Button(root, text="Predict", command=predict_image)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)


# Training buttons
save_banana_button = Button(root, text="Save as Banana", command=lambda: save_image("banana"))
save_banana_button.grid(row=5, column=0, pady=5)

save_apple_button = Button(root, text="Save as Apple", command=lambda: save_image("apple"))
save_apple_button.grid(row=5, column=1, pady=5)

# Clear button
clear_button = Button(root, text="Clear", command=clear_canvas)
clear_button.grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()