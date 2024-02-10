import numpy as np
import matplotlib.pyplot as plt


class PerceptronMachine():
    def __init__(self, Eta):
        self.Eta = Eta
        #self.bias = bias
        #self.N = N
        pass



    def forward(self, x, w, b):
        
        return np.sum(np.dot(x, w)) + b


    def train_GD(self, X, y):
        w = 1.1*np.ones((9,1))
        b = 0
       
        error = 1


        while error != 0:
            
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)

                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    w = w - self.Eta * np.sign(z) * X[i].T
                    b = b - self.Eta * np.sign(z)
            err = []
            for i in range(X.shape[0]):
                z = self.forward(X[i], w, b)
                sz = 1 if z >= 0 else 0
                if sz != y[i]:
                    err.append(1)
            error = sum(err)
        return w, b
    
    def sample_output(self, X, w, b):
        z = self.forward(X, w, b)
        sz = 1 if z >= 0 else 0
        return sz


    def test(self, X, y, w, b):
        err = []
        for i in range(X.shape[0]):
            z = self.forward(X[i], w, b)
            sz = 1 if z >= 0 else 0
            if sz != y[i]:
                err.append(1)
        error = sum(err)
        return error


################################################ Pattern recognition ##############################################
# Generating the X image?
x = 256* np.ones([3,3],dtype = int)
for i in range(3):
    x[i,i] = 0
    x[i,2-i] = 0
plt.imshow(x,cmap='gray')
# dataset preparation
np.random.seed(42)
# X pattern ([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
pattern = np.ones([3,3],dtype = int)
for i in range(3):
    pattern[i,i] = 0
    pattern[i,2-i] = 0

pattern = pattern.reshape(1,9)


N = 10 # number of training samples

dataset = []
label = []
for _ in range(N):
    random_matrix = np.random.randint(0, 2, size=(1,9))
    dataset.append(random_matrix)
    label.append(1 if np.array_equal(random_matrix, pattern) else 0)


random_pos = np.random.randint(0,N)
dataset.insert(random_pos, pattern)
label.insert(random_pos, 1)

# necessary
dataset = np.array(dataset)
label = np.array(label)
print(dataset.shape[0])



# test set preparation
np.random.seed(55)
N_test = 30 # number of training samples

dataset_test = []
label_test = []
for _ in range(N):
    random_matrix = np.random.randint(0, 2, size=(1,9))
    dataset_test.append(random_matrix)
    label_test.append(1 if np.array_equal(random_matrix, pattern) else 0)


random_pos_test = np.random.randint(0,N)
dataset_test.insert(random_pos, pattern)
label_test.insert(random_pos, 1)

# necessary
dataset_test = np.array(dataset)
label_test = np.array(label)
print(dataset_test.shape[0])


model = PerceptronMachine(0.1)
w_GD, b_GD = model.train_GD(dataset, label)
error = model.test(dataset_test, label_test, w_GD, b_GD)
print(error)

sample_output = model.sample_output(pattern, w_GD, b_GD)
print(sample_output)    
sample_output = model.sample_output(dataset_test[10], w_GD, b_GD)
print(sample_output) 
