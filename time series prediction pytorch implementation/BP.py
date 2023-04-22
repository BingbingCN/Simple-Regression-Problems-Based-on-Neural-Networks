import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

LSTM_train = pd.read_csv("1.csv", usecols=[1])
LSTM_test = pd.read_csv("2.csv", usecols=[1])
print(len(LSTM_train),len(LSTM_test))

LSTM_train = LSTM_train.astype('float')
LSTM_test = LSTM_test.astype('float')

# %%

time_window = 10
validation_size = 24
train_size = len(LSTM_train) - time_window
scaler = MinMaxScaler(feature_range=(0, 1))
fitted_transformer = scaler.fit(LSTM_train[:train_size + time_window])
train_transformed = fitted_transformer.transform(LSTM_train)

# %%

Xall, Yall = [], []
Xtestall, Ytestall = [], []

for i in range(time_window, len(LSTM_test)):
    Xtestall.append(train_transformed[i - time_window:i, 0])
    Ytestall.append(train_transformed[i, 0])

for i in range(time_window, len(train_transformed)):
    Xall.append(train_transformed[i - time_window:i, 0])
    Yall.append(train_transformed[i, 0])

Xall = np.array(Xall)
Yall = np.array(Yall)

Xtestall = np.array(Xall)
Ytestall = np.array(Yall)

Xtrain = Xall[:train_size, :]
Ytrain = fitted_transformer.inverse_transform(Yall[:train_size].reshape(-1,1))

Xvalidation = Xtestall[-validation_size:, :]
Yvalidation = fitted_transformer.inverse_transform(Ytestall[-validation_size:].reshape(-1,1))

# %%

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], time_window, 1))
Xvalidation = np.reshape(Xvalidation, (Xvalidation.shape[0], time_window, 1))

print(Xtrain.shape)
print(Xvalidation.shape)
print(Yvalidation.shape)
print(Ytrain.shape)

class TimeDataset(Dataset):
    def __init__(self,df_x,df_y):
        self.data=df_x
        self.label=df_y
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        data_value=torch.FloatTensor(self.data[index,:])
        label_value=torch.FloatTensor([self.label[index]])
        return data_value,label_value

Xtrain=Xtrain.reshape(-1,10)
Xvalidation=Xvalidation.reshape(-1,10)

train_dataset=TimeDataset(df_x=Xtrain,df_y=Ytrain)
valid_dataset=TimeDataset(df_x=Xvalidation,df_y=Yvalidation)
BATCH_SIZE=8
train_iterator=DataLoader(train_dataset,batch_size=BATCH_SIZE)
valid_iterator=DataLoader(valid_dataset,batch_size=BATCH_SIZE)
#test
for (data,label) in train_iterator:
    print(data)
    print(label)
    break



class FullyConnectedModel(nn.Module):
    def __init__(self,InputDim=10,OutputDim=1):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(InputDim,128),

            nn.Linear(128,128),
            nn.Linear(128,64),


            nn.Linear(64,16),

            nn.Linear(16,1)
        )

    def forward(self,data):
        return self.model(data)

def train(model, iterator, optimizer, criterion,device='cuda'):
    epoch_loss = 0
    model=model.to(device)
    model.train()
    for batch in iterator:
        batch[0]=batch[0].to(device)
        batch[1]=batch[1].to(device)
        criterion=criterion.to(device)
        optimizer.zero_grad()
        predictions = model(batch[0])
        loss = criterion(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator,criterion,device='cuda'):
    epoch_loss = 0
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            batch[0]=batch[0].to(device)
            batch[1]=batch[1].to(device)
            criterion=criterion.to(device)
            predictions = model(batch[0])
            loss = criterion(predictions, batch[1])
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lossfunction=nn.SmoothL1Loss()
fc_model=FullyConnectedModel(InputDim=10,OutputDim=1)
optimizer=torch.optim.Adam(fc_model.parameters(),lr=0.01)

N_epoch=50
train_loss_list=[]
valid_loss_list=[]
for i in range(N_epoch):
    train_loss=train(model=fc_model,iterator=train_iterator,criterion=lossfunction,optimizer=optimizer,device=device)
    valid_loss=evaluate(model=fc_model,iterator=valid_iterator,criterion=lossfunction,device=device)
    print("Epoch:",(i+1))
    print("Training Loss:",train_loss,"Valid Loss:",valid_loss)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    if i ==0:
        best_valid_loss=valid_loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print('Save Model..')
        torch.save(fc_model.state_dict(), 'fc_model.pt')

# fc_model.load_state_dict(torch.load('fc_model.pt'))
fc_model.to('cpu')
train_iterator=DataLoader(train_dataset,batch_size=1)
valid_iterator=DataLoader(valid_dataset,batch_size=1)
list1=[]
list2=[]
for batch in train_iterator:
    pre1=fc_model(batch[0])
    list1.append(pre1)
for batch in valid_iterator:
    pre2=fc_model(batch[0])
    list2.append(pre2)
list2=np.array([i.detach().numpy() for i in list2]).reshape(-1)
list1=np.array([i.detach().numpy() for i in list1]).reshape(-1)

Y_train_p=list1
Y_v_p=list2
Yall=fitted_transformer.inverse_transform(Yall.reshape(-1,1)).reshape(-1)
Yvalidation=Yvalidation.reshape(-1)
Yall=Yall.tolist()+Yvalidation.tolist()

def MSE(y,y_hat):
    return np.mean((y-y_hat) ** 2)
print("MSE is {:.4f}".format(MSE(Yvalidation,list2)))

plt.figure(figsize=(15,6))
plt.title('Underemployment Rate with BP')
plt.plot(range(len(Yall)),Yall,color='blue',label='Underemployment Rate')
plt.plot(range(len(Y_train_p)),Y_train_p,label='Underemployment Rate Predicted')
plt.plot(range(len(Y_train_p)+1,len(Y_train_p)+len(Y_v_p)+1),Y_v_p,label='BP Predicted')
# plt.plot(range(len(Y_train_p)+1,len(Y_train_p)+len(Yvalidation)+1),Yvalidation)
# plt.plot(validation.index,pd.DataFrame(LSTM_pred),color='red',label='LSTM Predicted')
plt.xlabel('Time')
plt.ylabel('Underemployment Rate')
plt.axvline(len(Y_train_p),color='black')
plt.legend(loc = "upper left")
plt.savefig('Prediction.jpg')