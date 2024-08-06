import matplotlib.pyplot as plt
import numpy as np
import random
from sys import exit

def plot_y(fig , y):
    for i in range(len(y)):
        dot = 'ko' if y[i]>0 else 'ro'
        plt.plot(X[i][0],X[i][1], dot)
    return fig

def plot(NN, X, Y):
    y = [i for j in Y for i in j]
    h = .02
    x_min = -0.8
    x_max = 1.2
    y_min = -0.8
    y_max = 1.2
    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)
    fig, ax = plt.subplots()
    fig = plot_y(fig, y)
    Z = []
    for x in xx:
        z = []
        for y in yy:
            out = NN.forward([x,y])[0][0]
            if out > 0.5:
                z.append(1)
            else:
                z.append(0)
        Z.append(z)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis([-0.2,1.2,-0.2,1.2])
    # plt.savefig('./'+str(iteration))
    plt.show()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def softmax(self, Z):
        if np.mean(Z) > 0.5:
            return np.array([[1]])
        else:
            return np.array([[0]])
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2
    
    def backward(self, X, Y, output):
        m = X.shape[0]
        
        dZ2 = output - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, Y, epochs=10001):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, Y, output)
            if epoch % 1000 == 0:
                loss = -np.mean(Y * np.log(output))
                print(f'Epoch {epoch}, Loss: {loss}')
                plot(self, X, Y)
    

X = np.array([[0,0],[0,1],[1,0],[1,1],[0.5,0.5]])
Y = np.array([[0],[1],[1],[1],[0]])
nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1)
nn.train(X, Y)

print(nn.forward(X))