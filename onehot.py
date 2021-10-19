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

time0 = time()
path = './machine_learning.csv'
file = pd.read_csv(path)
file = file.dropna(axis=0,how='any')
y = file.iloc[:,-1]
data = []
for i in y:
    if i <=3.4 and i >=3:
        data.append('4')
    elif i>=2.7 and i<3:
        data.append('3')
    elif i<2.7 and i>=2.5:
        data.append('2')
    elif i<2.5 and i>=1.8:
        data.append('1')
    elif i< 1.8:
        data.append('0')
    else:
        data.append('5')
#print(data)
data = np.array(data).reshape(-1,1)
#print(data.shape)
statisfy = ['no','far under satisfy','almost satisfy', 'satisfy', 'over satisfy']
#statisfy = np.array(statisfy)

enc=OneHotEncoder()
enc.fit(data)
#print(enc.categories_)
y = enc.fit_transform(data).toarray()

#print(y)
X = file.iloc[:,:-1]
X = np.array(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X = torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
#X = torch.reshape(X,(-1,160,6))
y = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

"""X = torch.FloatTensor(X_train).to('cuda:0')
X_test=torch.FloatTensor(X_test).to('cuda:0')
#X = torch.reshape(X,(-1,160,6))
y = torch.FloatTensor(y_train).to(X_test.device)
y_test = torch.FloatTensor(y_test).to(X_test.device)
#print(X.shape)"""
#total_data = torch.cat((X,y),1)

#print(X.shape,y.shape)

#创建模型

class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.dis = nn.Sequential(nn.LayerNorm(6),
                                 nn.Linear(6, 15),
                                 nn.ReLU(),
                                 nn.Linear(15, 20),
                                 nn.ReLU(),
                                 nn.Linear(20, 15),
                                 nn.ReLU(),
                                 nn.Linear(15,6),
                                 nn.ReLU())

    def forward(self,input):
        input = self.dis(input)
        return input

test_model = test_model()
if torch.cuda.is_available():
    test_model=test_model.cuda()
loss_value = []
criteria = nn.MSELoss()
if torch.cuda.is_available():
    criteria=criteria.cuda()
test_model_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.0005,betas=(0.5,0.999))
epochs=1000
for epoch in range(epochs):
    running_loss =0.0
    #running_loss_detach=0.0
    for i, data in enumerate(X):
        if torch.cuda.is_available():
            data = data.cuda()
            y_cuda = y[i,:].cuda()
        input_1 = test_model(data)
        loss = criteria(input_1,y_cuda)
        test_model_optimizer.zero_grad()
        loss.backward()
        test_model_optimizer.step()
        #loss_detach = loss.to(torch.device('cpu'))
        #lose_detach = loss.detach().cpu().numpy()
        running_loss = running_loss + loss
        #running_loss_detach = running_loss_detach + lose_detach
    running_loss = running_loss.detach().cpu().numpy()
    print(running_loss)
    loss_value.append(running_loss)
print('The time-consuming: %s'% datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

"""total_input=[]
y_value=[]
for i,test in enumerate(X_test):
    input_2 = test_model(test)
    loss = criteria(input_2,y_test[i,:])
    input_2 = input_2.detach().cpu().numpy()
    y_test_value = y_test[i,:].detach().cpu().numpy()
    total_input.append(input_2)
    y_value.append(y_test_value)


    #print(input_2,y_test[i,:])
predict = pd.DataFrame(total_input)
#predict.columns=['predict']
predict['true_value']=y_value
predict.to_csv('./validation_1.csv')
print('The time-consuming: %s'% datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
#print(test_model.state_dict()['dis.1.weight'])
#print(test_model.state_dict()['dis.3.weight'])
#print(test_model.state_dict()['dis.5.weight'])
#看weight
#writer = SummaryWriter('./logs_1')
#writer.add_graph(test_model,X)
#writer.close()

plt.plot(range(epochs),loss_value,label='loss_value')
plt.legend()
plt.show()"""

