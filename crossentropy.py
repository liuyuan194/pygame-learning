import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import DataLoader
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
import datetime
from torch.utils.tensorboard import SummaryWriter


# Visualize the data and precision

writer = SummaryWriter('./logs_train')
#Loading data
time0 = time()
path = './machine_learning.csv'
file = pd.read_csv(path)
file = file.dropna(axis=0,how='any')
y = file.iloc[:,-1]

# Label VFE
file['class'] = [0 if vfe < 2.5 else 1 for vfe in y]


# transform to np
X = file.iloc[:,:-2].to_numpy()
#X = np.array(X)
y = file['class']
y = np.array(y)
y = np.reshape(y,(-1,1))
#splitting data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#transform to tensor

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

#build model

class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.dis = nn.Sequential(nn.LayerNorm(6),
                                 nn.Linear(6, 64),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(64,1),
                                 nn.Sigmoid()

                        )

    def forward(self,input):
        input = self.dis(input)
        return input


# Sample model
test_model = test_model()


loss_value = []

# Cross entropy as loss function
criteria = nn.BCELoss()

# Make optimizer
test_model_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.0003,betas=(0.5,0.999))
epochs=1000


# Testing accurancy

def test_sample_accuracy(test):
    total_accuracy = []
    for i,test in enumerate(test):
        input_2 = test_model(test)
        #loss = criteria(input_2,y_test[i])
        if input_2 >= 0.5:                  # theshold = 0.5
            prediction = 1
        else:
            prediction = 0
        accuracy = prediction == y_test[i]
        total_accuracy.append(accuracy)

    return sum(total_accuracy)/len(total_accuracy)

# training

total_train_step=0

for epoch in range(epochs):
    running_loss =0.0
    for i, data in enumerate(X_train):
        input_1 = test_model(data)
        loss = criteria(input_1,y_train[i])
        test_model_optimizer.zero_grad()
        loss.backward()
        test_model_optimizer.step()
        running_loss = running_loss + loss
    print(running_loss)
    test_accuracy = test_sample_accuracy(X_test)   # accuracy of each step model
    total_train_step += 1
    loss_value.append(running_loss)
    writer.add_scalar('train_loss_4', running_loss.item(), total_train_step)  # wrting to tensorboard
    writer.add_scalar('test_accuracy_4', test_accuracy.item(), total_train_step)

print(test_accuracy)    # the last step accuracy
print('The time-consuming: %s'% datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))