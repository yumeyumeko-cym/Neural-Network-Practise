import torch
import torchvision.datasets as DS
import torchvision.transforms as transform
from tqdm import tqdm
import FNN

mnist = DS.MNIST('./data',train=True,transform=transform.ToTensor(),download=True)
#print(dir(mnist))
print(len(mnist))

mnist_data_copy = mnist.data.clone().detach()
mnist_targets_copy = mnist.targets.clone().detach()

#print(mnist_copy[0])
#print(mnist.data[0])
#flag = torch.where(mnist.data[0] == mnist_copy[0], 1, 0)
#print(flag)

mnist_data_copy = mnist_data_copy.reshape(60000, 784)
torch.manual_seed(42)
x = mnist_data_copy.float() ## complete
v = mnist_targets_copy.int() ## complete


# define the model 
model = FNN.myFNN()
grad_1 = []

# for input data-point x and label v pass forward
for i in tqdm(range(len(x))):
    model.forward(x[i],v[i])
    model.backward()
    grad_1.append(model.grad_1)


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

print("\n model.grad2: ", model.grad_2.shape)
print("\n model.grad1: ", grad_1[-1])
