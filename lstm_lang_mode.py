import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from datasets import Dataset
from transformers import get_scheduler
import os
import random
import numpy as np
from cp_feature_extractor import model_output
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
#set seed
torch.manual_seed(0)
import wandb
# 1. Start a W&B run
wandb.init(project='lstm_langmodel')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 6
num_epochs = 500
batch_size = 32
config = wandb.config
config.batch_size = batch_size
# 2. Save model inputs and hyperparameters
config = wandb.config
learning_rate = 0.001
config.learning_rate = learning_rate

input_size = 1
sequence_length = 560
hidden_size = 64
config.hidden_size = hidden_size
num_layers = 2
config.num_layers  = num_layers

# MNIST dataset 
def data_extract( dataset_path): 
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")
    accuracy = 0
    num_files = 0
    pred_languages = []
    actual_languages= []
    cf_prb_m = {}
    #empty data frame with column names audio_embedding and language
    df_lang = pd.DataFrame(columns=['audio_embedding', 'language'])
    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == "test.csv":
                #reading the csv file
                df = pd.read_csv(subdir + "/" + file)
                #read the first column of the csv file
                df = df.iloc[:, 0].values
                for filename in df[:1000]:
                    audio_path = subdir + "/wav/" + filename.split('.')[0]+".wav"
                    codebook = 1
                    mdl_out_cd_1 = model_output(audio_path = audio_path,codebook = codebook,feature_extractor = feature_extractor,model=model)
                    codebook = 2
                    mdl_out_cd_2 = model_output(audio_path = audio_path,codebook = codebook,feature_extractor = feature_extractor,model=model)
                    mdl_out = mdl_out_cd_1+np.array(np.array(mdl_out_cd_2)+320).tolist()
                    label = subdir.split('/')[-1]
                    #append the audio embedding and language to the dataframe
                    df_lang = df_lang.append({'audio_embedding': torch.tensor(mdl_out), 'language': label}, ignore_index=True)
    return df_lang
lang = ["de","en","es","fr","it","ru"]
df =  pd.DataFrame(columns=['audio_embedding', 'language'])
#pickle file exists
if os.path.exists("lang_data.pkl"):
    df = pd.read_pickle("lang_data.pkl")
else:
    for l in lang:
        df_lang = data_extract(dataset_path = f"/corpora/common_phone/{l}")
        #append df_lang to df
        breakpoint()
        df = df.append(df_lang, ignore_index=True)
    #save the dataframe to pickle file
    df.to_pickle("lang_data.pkl")
#dataset from dataframe

label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(lang):
    label2id[label] = torch.tensor(i)
    id2label[torch.tensor(i)] = label
    label2id_int[label] = i
#get the audio embedding and language from the dataframe and convert it to list
audio_embedding = df['audio_embedding'].tolist()
#ValueError: only one element tensors can be converted to Python scalars
#convert the list with each element of different sizes to torch tensor

language = df['language'].tolist()


language = [label2id[l] for l in language]
language = torch.tensor(language)
# #create one hot encoding for the language
# language = torch.nn.functional.one_hot(language, num_classes=6)
#convert dtype of language to torch.float32
language = language.type(torch.float32)

#split the audio_embedding language and into train and test
train_dataset_x, test_dataset_x,train_dataset_y, test_dataset_y = train_test_split(audio_embedding, language, test_size=0.2, random_state=42)
#dataset for train and test
print(len(train_dataset_x))
print(len(test_dataset_x))
# train_dataset = Dataset(train_dataset_x,train_dataset_y)
# test_dataset = Dataset(test_dataset_x,test_dataset_y)

# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
best_acc = 0
lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_epochs*(len(train_dataset_y)/batch_size)*0.1),
        num_training_steps=num_epochs*(len(train_dataset_y)/batch_size),
    )
c = list(zip(train_dataset_x, train_dataset_y))
# Train the model
n_total_steps = len(train_dataset_y)
for epoch in range(num_epochs):
    random.shuffle(c)
    train_dataset_x, train_dataset_y = zip(*c)
    loss_epoch =0
    loss_batch = 0
    for i, labels in enumerate(train_dataset_y):  
        # origin shape: [N, 1, 28, 28]
        # resized: [N, 28, 28]
        images = torch.tensor(train_dataset_x[i],dtype=torch.float32).reshape(1,-1,1).to(device)
        labels= labels.unsqueeze(0)
        labels= labels.type(torch.LongTensor).to(device)
        # Forward pass
    
        outputs = model(images)
        loss = criterion(outputs,labels )
        
        #accumaulate the loss into a variable for printing after each epoch
        loss_epoch += loss.item()
        loss_batch += loss
        if (i+1) % batch_size == 0:
                   
            # Backward and optimize
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            lr_scheduler.step()
            loss_batch = 0 
        
        if (i+1) % 1000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    wandb.log({"loss": loss_epoch/len(train_dataset_y)})
    print("loss per epoch",loss_epoch/len(train_dataset_y))
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i, labels in enumerate(test_dataset_y):
            images =torch.tensor(test_dataset_x[i],dtype=torch.float32).reshape(1,-1,1).to(device)
            labels = labels.to(device)
            labels= labels.type(torch.LongTensor)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += 1
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        wandb.log({"validation accuracy": acc})
        print(f'Accuracy of the network on the test images: {acc} %')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pt')
            print('model saved')
            print(f'Best Accuracy of the network on the test images: {best_acc} %')