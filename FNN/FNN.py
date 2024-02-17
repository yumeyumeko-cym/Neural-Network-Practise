import torch
import numpy as np

class hidden():

    def __init__(self, input_size, output_size, W):
        # define the attributes of this class 
        self.input_size = input_size
        self.output_size = output_size
        self.weights = W

    def MyReLU(self, x):
        '''
        x is the input tensor and out is the ouput tensor.
        '''
        out = torch.max(x, torch.tensor(0))
        return out

    def relu_derivative(self, x):

        return torch.where(x > torch.zeros(x.shape), 1.0, 0.0)
    def forward(self, x):
        '''
        x is input to the hidden layer
        x is of size input_size + 1
        '''
        self.z = torch.matmul(self.weights, x)
        self.y = self.MyReLU(self.z)
        self.y = torch.cat((torch.tensor([1]), self.y)) # Add 1 as the fisrt entry of self.y and build a new self.y with size output_size + 1
        
        return self.z, self.y
    
    def backward(self, g_y):
        '''
        g_y is gradient w.r.t. output
        g_y is of size output_size
        '''
        #print("self.z: ", self.z.shape)
        d_relu = self.relu_derivative(self.z)
        #print("d_relu: ", d_relu.shape)
        #print("g_Y: ", g_y.shape)
        self.g_z = g_y * d_relu
        #print("g_z: ", self.g_z.shape)
        self.g_x = torch.matmul(self.g_z, self.weights)
        self.g_x = self.g_x[1:]
        
        # self.g_x should be of size input_size
        return self.g_z, self.g_x
    

class outLayer():
    def __init__(self, input_size, output_size, W):
        # define the attributes of this class 
        self.input_size = input_size # complete
        self.output_size = output_size # complete
        # initiate the matrix of weights
        self.weights = W


    def softmax(self, x):
        """
        Compute the softmax function for an n-dimensional input tensor.
        """
        softmax_provided = torch.nn.Softmax()
        return softmax_provided(x)
    
    
    def softmax_grad(self, s):
        """
        Compute the derivative of the softmax function.
        Input `s` is the softmax value of the original input `x`.
        `s.shape = (batch_size, num_classes)`
        """
        # Reshape the 1-D softmax to 2-D for matrix multiplication
        s = s.reshape(-1, 1)
        # Compute the Jacobian matrix
        jacobian_m = torch.diagflat(s) - torch.mm(s, s.T)
        return jacobian_m

    def MyCrossEntropyLoss(self, y, v):
        '''
        y is the output tensor and v is the true label
        v = {0, 1, ..., 9}
        out is the loss
        '''
        out = -torch.log(y[v])
        return out   


    def forward(self, x):
        '''
        x is output of last hidden layer
        x is of size input_size + 1
        '''
        self.z = torch.matmul(self.weights, x) # complete
        
        #self.y = torch.nn.functional.softmax(self.z, dim=0)
        self.y = self.softmax(self.z)


        # size of self.y should be # of classes
        return self.z, self.y
    

    def cal_loss(self, v):
        '''
        v is the true label {0, 1, ..., 9}
        '''
        self.loss = self.MyCrossEntropyLoss(self.y, v.int()) # complete
        # self.loss is cross-entropy between self.y and v
        self.g_y = -v/self.y + (1-v)/(1-self.y) # complete
        #  self.g_y is the gradient of loss w.r.t. output
        #  self.g_y is of size output_size
        return self.loss, self.g_y
    

    
    def backward(self):
        self.g_z = torch.matmul(self.softmax_grad(self.y), self.g_y) # complete
        self.g_x = torch.matmul( self.g_z, self.weights) # complete
        self.g_x= self.g_x[1:]
        # self.g_x should be of size input_size (remember to drop the first entry after calculation.)
        # No input needed as loss generated self.g_y
        return self.g_z, self.g_x



class myFNN():
    def __init__(self):
        # define the attributes of this class 
        self.input_size = 784
        
        weights_1 = 0.001*torch.rand(128, 784+1) # complete
        self.hidden_size_1 = 128
        self.hidden1 = hidden(784, 128, weights_1)
        
        weights_2 = 0.01*torch.rand(128, 128+1) # complete
        self.hidden_size_2 = 128
        self.hidden2 = hidden(128, 128, weights_2)
        
        weights_3 = torch.rand(10, 128+1) # complete



        self.num_classes = 10
        self.outLayer = outLayer(128, 10, weights_3)
    def forward(self, x, v):

        # add dummy 1 to x at index 0
        dummy = torch.tensor([1], device=x.device, dtype=x.dtype)

        x = torch.cat((dummy,x), axis=0) # complete
        # forward pass through hidden layer 1 
        self.hidden1.forward(x)
        # forward pass through hidden layer 2
        self.hidden2.forward(self.hidden1.y)
        # forward pass through output layer
        self.outLayer.forward(self.hidden2.y)
        # complete
        #print(v)
        self.outLayer.cal_loss(v)
        # complete
        return
    def backward(self, x):
        # backward pass through output layer
        self.outLayer.backward()
        #print("output_g_z shape",self.outLayer.g_z.shape)
        # backward pass through hidden layer 2
        self.hidden2.backward(self.outLayer.g_x)
        # backward pass through hidden layer 1
        self.hidden1.backward(self.hidden2.g_x)


        dummy = torch.tensor([1], device=x.device, dtype=x.dtype)
        x_ = torch.cat((dummy,x), axis=0) # complete       
        # Now, compute gradients w.r.t. weights
        self.grad_3 = torch.outer(self.outLayer.g_z, self.hidden2.y) # complete < gradient for output layer>
        
        self.grad_2 = torch.outer(self.hidden2.g_z, self.hidden1.y) # complete < gradient for layer 2>

        self.grad_1 = torch.outer(self.hidden1.g_z, x_) # complete < gradient for layer 1>
    

torch.manual_seed(42)
x = torch.rand(784) ## complete
#print(x)
v = torch.tensor([0]) ## complete


# define the model 
model = myFNN()
# for input data-point x and label v pass forward
model.forward(x,v)
model.backward(x)

#print(model.hidden1.y)
#print(model.hidden2.y)
print(model.hidden2.y.shape)
#print(model.outLayer.weights.shape)
#print(model.outLayer.y)
print(model.outLayer.y.shape)
print(model.outLayer.loss)
print(model.outLayer.g_y)
#print("\n model.outLayer.g_z: ", model.outLayer.g_z.shape)
#print("\n model.outLayer.g_x: ", model.outLayer.g_x.shape)

print("\n model.grad2: ", model.grad_3.shape)
print("\n model.grad2: ", model.grad_2.shape)
print("\n model.grad1: ", model.grad_1.shape)