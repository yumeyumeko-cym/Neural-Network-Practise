import numpy as np
import matplotlib.pyplot as plt

class PerceptronMachine():
    def __init__(self, Eta):
        self.Eta = Eta
        #self.bias = bias
        #self.N = N
        pass



    def forward(self, x, w, b):
        return np.dot(x, w) + b

    def train(self, X, y):
        w = [1.1,1.1]
        b = 0
       
        error = 1


        while error != 0:
            
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)

                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    w = w - np.sign(z) * X[i]
                    b = b - np.sign(z)
            err = []
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)
                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    err.append(1)
            error = sum(err)
        return w, b

    def train_GD(self, X, y):
        w = [1.1,1.1]
        b = 0
       
        error = 1


        while error != 0:
            
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)

                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    w = w - self.Eta * np.sign(z) * X[i]
                    b = b - self.Eta * np.sign(z)
            err = []
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)
                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    err.append(1)
            error = sum(err)
        return w, b
    

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
Eta = 0.1 # learning rate
model = PerceptronMachine(Eta)
w, b = model.train(X, y)
w_GD, b_GD = model.train_GD(X, y)
print(w, b)
print("Weights:", w)
print("Bias:", b)



# Test the perceptron on the dataset
print("Testing the trained perceptron:")
for i in range(X.shape[0]):
    output = model.forward(X[i], w, b)
    prediction = 1 if output >= 0 else 0
    print(f"Input: {X[i]}, Predicted: {prediction}, Actual: {y[i]}")
print("***********************************************************************")
print(w_GD, b_GD)
print("Weights (Eta):", w_GD)
print("Bias (Eta):", b_GD)
print(f"Testing the trained perceptron (Eta={Eta})")
for i in range(X.shape[0]):
    output = model.forward(X[i], w_GD, b_GD)
    prediction = 1 if output >= 0 else 0
    print(f"Input: {X[i]}, Predicted: {prediction}, Actual: {y[i]}")