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

writer = SummaryWriter('./logs_train')
time0 = time()
path = './machine_learning.csv'
file = pd.read_csv(path)
file = file.dropna(axis=0,how='any')
y = file.iloc[:,-1]
data = []

for i in y:
    if i < 2.5:
        data.append(0)
    else:
        data.append(1)
#print(data)
data = np.array(data).reshape(-1,1)

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
class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        self.dis = nn.Sequential(nn.LayerNorm(6),
                                 nn.Linear(6, 15),
                                 #nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(15, 20),
                                 #nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(20, 15),
                                 #nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(15,6),
                                 #nn.Dropout(0.5),
                                 nn.ReLU(),
                                 nn.Linear(6,2),
                                 nn.Softmax(dim=0))

    def forward(self,input):
        input = self.dis(input)
        return input

test_model = test_model()
#if torch.cuda.is_available():
    #test_model=test_model.cuda()
loss_value = []
criteria = nn.CrossEntropyLoss()
#if torch.cuda.is_available():
    #criteria=criteria.cuda()
test_model_optimizer = torch.optim.SGD(test_model.parameters(), lr=0.0004)
epochs=800

total_train_step=0
total_test_step = 0
def test_sample_accuracy(test):
    total_accuracy = []
    for i,test in enumerate(test):
        input_2 = test_model(test)
        loss = criteria(input_2,y_test[i,:])
        #print(input_2,y_test[i,:])
        accuracy = input_2.argmax(0)==y_test[i,:].argmax(0)
        total_accuracy.append(accuracy)

    return sum(total_accuracy)/len(total_accuracy)



total_accuracy = []


for epoch in range(epochs):
    running_loss =0.0
    #running_loss_detach=0.0
    for i, data in enumerate(X):
        #if torch.cuda.is_available():
            #data = data.cuda()
            #y_cuda = y[i,:].cuda()
        input_1 = test_model(data)
        loss = criteria(input_1,y[i,:])
        test_model_optimizer.zero_grad()
        loss.backward()
        test_model_optimizer.step()
        #loss_detach = loss.to(torch.device('cpu'))
        #lose_detach = loss.detach().cpu().numpy()
        running_loss = running_loss + loss
        #running_loss_detach = running_loss_detach + lose_detach
    #running_loss = running_loss.detach().numpy()
    print(running_loss)
    test_accuracy = test_sample_accuracy(X_test)   # accuracy of each step model
    total_train_step += 1
    loss_value.append(running_loss)
    #total_accuracy.append(test_accuracy)
    writer.add_scalar('train_loss_3',running_loss.item(),total_train_step)
    writer.add_scalar('test_accuracy_3', test_accuracy.item(), total_train_step)

print('The time-consuming: %s'% datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

total_input=[]
y_value=[]




    #input_2 = input_2.detach().numpy() #准备数据
    #y_test_value = y_test[i,:].detach().numpy()
    #total_input.append(input_2)
    #y_value.append(y_test_value)
#predict = pd.DataFrame(total_input) #写入表格
#predict['true_value']=y_value
#print(test_model.state_dict()['dis.1.weight'])#查看权重
#print(test_model.state_dict()['dis.3.weight'])
#print(test_model.state_dict()['dis.5.weight'])
#print(test_model.state_dict()['dis.9.weight'])
#predict.to_csv('./validation_2.csv')
#plt.plot(range(epochs),loss_value,label='loss_value')#画图
#plt.legend()
#plt.show()