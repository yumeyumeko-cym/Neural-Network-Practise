import numpy as np
import matplotlib.pyplot as plt

class LinearMachine():
    def __init__(self):
        pass
    
    def data_synthesizer(self, I, v0, sigma_v2, h0, sigma_h2):
        """
        Generates data for the linear machine.
        """
        v = np.abs(np.random.normal(v0, np.sqrt(sigma_v2), I))
        h = np.abs(np.random.normal(h0, np.sqrt(sigma_h2), I))
        d = 0.45 * v * np.sqrt(h)
        
        # Stack v, h, and d horizontally to form a data matrix X
        X = np.column_stack((v, h, d))
        # X.shape = (I,3)


        return X
    
    def train_GD(self, data, alpha, err_tolerance):
        X = data[:, [0, 1]]  # v and h, Ix2
        y = data[:, 2]  # d, Ix1
        y = y.reshape(len(y), 1)
        I = len(data[:, 0])

        deviation = 100
        # Initialize weight with shape (2, 1)
        w = np.ones((2, 1))
        err_old = 100
        max_epochs = 100000  # Define a maximum number of iterations to avoid infinite loop
        for _ in range(max_epochs):
            # Compute prediction
            pred = np.dot(X, w)
            # Calculate loss
            loss = pred - y
            # Compute gradient
            gradient = 2 / I * np.dot(X.T, loss)
            # Update weight
            w = w - alpha * gradient
            # Calculate deviation
            err = 1 / I * np.dot(loss.T, loss)

            deviation = abs(err - err_old)
            err_old = err

            # Check convergence
            if deviation <= err_tolerance:
                break
            #print(w)
        return w


    def train(self, data):
        X = data[:,[0,1]] # v and h, Ix2
        y = data[:,2] # d, Ix1
        y = y.reshape(len(y),1)
        #print(X.shape)
        #print(y.shape)
        I = len(data[:,0])
        
        
        
        w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
        
        #print(gradient)
        return w   

    def test(self, data, w):
        X = data[:,[0,1]] # v and h, Ix2
        y = data[:,2] # d, Ix1
        y = y.reshape(len(y),1)

        I = len(data[:,0])
        
        pred = np.dot(X,w)
        loss = pred - y
        err = 1/I*np.dot(loss.T,loss)

        return err


def numerical_eval(T, I, J, v0, sigma_v2, h0, sigma_h2):
    model = LinearMachine()
    errors = []
    for _ in range(T):
        X_train = model.data_synthesizer(I, v0, sigma_v2, h0, sigma_h2)
        X_test = model.data_synthesizer(J, v0, sigma_v2, h0, sigma_h2)

        w_trained = model.train_GD(X_train, 0.001, 0.00)
        #w_trained = model.train(X_train)
        err_trained = model.test(X_test, w_trained)
        errors.append(err_trained)


    err = np.mean(errors)
    return err    

def numerical_eval_true(T, I, J, v0, sigma_v2, h0, sigma_h2):
    model = LinearMachine()
    errors = []
    for _ in range(T):
        X_train = model.data_synthesizer(I, v0, sigma_v2, h0, sigma_h2)
        X_test = model.data_synthesizer(J, v0, sigma_v2, h0, sigma_h2)

        
        w_trained = model.train(X_train)
        err_trained = model.test(X_test, w_trained)
        errors.append(err_trained)


    err = np.mean(errors)
    return err    


def compare_weights(model, data, learning_rates, err_tolerance):
    distances = []
    # Train the model using the 'train' method to get the analytical solution
    w_train = model.train(data)
    
    for alpha in learning_rates:
        # Train the model using the 'train_GD' method
        w_gd= model.train_GD(data, alpha, err_tolerance)
        
        # Compute Euclidean distance between weights
        distance = np.linalg.norm(w_train - w_gd)
        distances.append(distance)
    
    return distances

# Generate synthetic data
model = LinearMachine()
X = model.data_synthesizer(100, 1, 5, 3, 3)

# Define learning rates and error tolerance
learning_rates = np.logspace(-8, -2, 10, base=10)
err_tolerance = 0.001

# Compare weights for different learning rates
distances = compare_weights(model, X, learning_rates, err_tolerance)

# Plotting the differences
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, distances, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Euclidean Distance Between Weights')
plt.title('Distance Between Weights from train and train_GD')
plt.grid(True)
plt.show()


# if the initial learning rate is too small, the algorithm tends to trap in local minima.



I = np.linspace(100, 1000, 10, dtype=int)
empirical_err = []
empirical_err_true = []
for i in I:
    error = numerical_eval(100, i, 10, 1, 5, 3, 3)
    error_true = numerical_eval_true(100, i, 10, 1, 5, 3, 3)
    empirical_err.append(error)
    empirical_err_true.append(error_true)

plt.figure(figsize=(10, 6))

# Plot empirical_err with a red line
plt.plot(I, empirical_err, marker='o', color='red', label='Empirical Error')


plt.plot(I, empirical_err_true, marker='o', color='blue', label='Empirical Error with Calculated Weights')

plt.xlabel('Data Size I')
plt.ylabel('Average Empirical Error')
plt.title('Average Empirical Error vs. Data Size')
plt.grid(True)
plt.legend() 
plt.show()
