# We import torch and the nn module
import torch
import torch.nn as nn

# We import NumPy and tqdm
import numpy as np
from tqdm import tqdm

# We need few items from Scikit-Learn
import sklearn.datasets as DataSets
from sklearn.model_selection import train_test_split

# Also we need to plot few curves
import matplotlib.pyplot as plt




dataset = DataSets.make_classification(n_samples = 10000
                                       , n_features = 20
                                       , n_informative=5
                                       , n_redundant=15
                                       , random_state=1)
X, v = dataset

def data_splitter(X, v, batch_size, train_size):
    '''
    X is list of data-points
    v is list of labels
    train_size is the fraction of data-points used for training
    '''
    X_train, X_test, v_train, v_test = train_test_split(X, v, train_size=train_size, shuffle = True, random_state=1)
    X_train = torch.tensor(X_train,dtype=torch.float32)
    v_train = torch.tensor(v_train,dtype=torch.float32).reshape(-1,1)
    X_test = torch.tensor(X_test,dtype=torch.float32)
    v_test = torch.tensor(v_test,dtype=torch.float32).reshape(-1,1)
    
    batch_indx = torch.arange(0,X_train.shape[0],batch_size)
    return X_train, v_train, X_test, v_test, batch_indx

class myClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(20, 64)
        self.active1 = nn.ReLU()

        self.layer2 = nn.Linear(64, 32)
        self.active2 = nn.ReLU()

        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.active1(self.layer1(x))
        x = self.active2(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x
    

def training_loop(model):
    # define the loss and optimizer
    loss_fn = nn.BCELoss() # binary cross-entropy
    optimizer = torch.optim.Adam(model.parameters()
                            , lr=0.0001 # this specifies the learning rate
                            )

    # set the training parameters
    n_epochs = 300   # number of epochs
    batch_size = 40  # batch size

    # specify training and test datasets and the batch indices
    # use data_splitter() and X, v are generated by Scikit-Learn
    X_train, v_train, X_test, v_test, batch_indx = data_splitter(X, v, batch_size, 0.8)


    # make empty list to save training and test risk
    train_risk = []
    test_risk = []

    # training loop

    # we visualize the training progress via tqdm
    with tqdm(range(n_epochs), unit="epoch") as epoch_bar:
        epoch_bar.set_description("training loop")
        for epoch in epoch_bar:

            # tell pytorch that you start training
            model.train()

            for indx in batch_indx:
                # take a batch of samples
                X_batch = X_train[indx:indx+batch_size-1]
                v_batch = v_train[indx:indx+batch_size-1]

                # pass forward the mini-batch
                y_batch = model(X_batch)

                # compute the loss
                loss = loss_fn(y_batch, v_batch)

                # backward pass
                # first make gradient zero
                optimizer.zero_grad()
                # then, compute the gradient of loss
                loss.backward()

                # now update weights by one optimization step
                optimizer.step()

            # we are done with one epoch
            # we now evaluate training and test risks
            # first we tell pytorch we are doing evaluation
            model.eval()

            # now we evaluate the training risk
            y_train = model(X_train)
            CE_train = loss_fn(y_train, v_train)
            train_risk.append(CE_train.item())

            # then we evaluate the test risk
            y_test = model(X_test)
            CE_test = loss_fn(y_test, v_test)
            test_risk.append(CE_test.item())
        return train_risk, test_risk


myModel = myClassifier()
train_risk, test_risk = training_loop(myModel)

# complete <plot training risk>
# complete <plot test risk>
# Plot training risk

plt.plot(train_risk, label='Training Risk')

# Plot test risk
plt.plot(test_risk, label='Test Risk')

# Add title and labels to the plot
plt.title('Training and Test Risk over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Risk')

# Show legend
plt.legend()

# Display the plot
plt.show()

