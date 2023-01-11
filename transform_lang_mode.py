import torch
import torch.nn as nn
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
from transform import TransformerModel
#set manual seed for the packages
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(0)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
import wandb
# 1. Start a W&B run
wandb.init(project='transformer_batch')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 



config = wandb.config
config.batch_size = 2
config.num_epochs = 500
# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 1e-4
# MNIST dataset 
lang = ["de","en","es","fr","it","ru"]
num_classes = len(lang)
model_checkpoint = "facebook/wav2vec2-large-xlsr-53"
label2id, id2label,label2id_int = dict(), dict(),dict()
for i, label in enumerate(lang):
    label2id[label] = torch.tensor(i)
    id2label[torch.tensor(i)] = label
    label2id_int[label] = i
def data_extract(model_checkpoint, dataset_path_m,file_to_extract): 
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = Wav2Vec2ForPreTraining.from_pretrained(model_checkpoint)
    for l in lang:
        dataset_path = dataset_path_m + l + "/"
        #read the csv file first column using the path dataset_path + "/" + file_to_extract to get the path of the audio file ignoreing the first row
        for audio_file_path in pd.read_csv(dataset_path + "/" + file_to_extract, header=None)[0][1:]:
            audio_path = dataset_path + "/wav/" + audio_file_path.split('.')[0]+".wav"
            codebook = 1
            mdl_out_cd_1 = model_output(audio_path = audio_path,codebook = codebook,feature_extractor = feature_extractor,model=model)
            codebook = 2
            mdl_out_cd_2 = model_output(audio_path = audio_path,codebook = codebook,feature_extractor = feature_extractor,model=model)
            mdl_out = mdl_out_cd_1+np.array(np.array(mdl_out_cd_2)+320).tolist()
            #yield dictioanry with the audio feature and the label
            yield {"input_values":torch.tensor(mdl_out).unsqueeze(dim = 0),"labels":torch.tensor(label2id[l]).unsqueeze(dim = 0)}
#load_from_disk if train_dataset contains the data
if os.path.exists("train_dataset")and os.path.exists("val_dataset") :
    train_dataset = Dataset.load_from_disk("train_dataset")
    val_dataset = Dataset.load_from_disk("val_dataset")
    # test_dataset = Dataset.load_from_disk("test_dataset")
else:
    file_to_extract = "train.csv"
    # Create a new function that always uses the value 'hello' for arg1
    train_dataset= Dataset.from_generator(data_extract,gen_kwargs={"model_checkpoint":model_checkpoint,"dataset_path_m":f"/corpora/common_phone/","file_to_extract":file_to_extract},cache_dir = "/cache")
                                        #,output_types={"input_values":torch.tensor,"labels":torch.tensor})
    # file_to_extract = "dev.csv"
    # val_dataset = Dataset.from_generator(data_extract,gen_kwargs={"model_checkpoint":model_checkpoint,"dataset_path_m":f"/corpora/common_phone/","file_to_extract":file_to_extract},cache_dir = "/cache")
    file_to_extract = "test.csv"
    val_dataset = Dataset.from_generator(data_extract,gen_kwargs={"model_checkpoint":model_checkpoint,"dataset_path_m":f"/corpora/common_phone/","file_to_extract":file_to_extract},cache_dir = "/cache")
    #save the dataset in optimal format by using the save_to_disk function with folders created
    #create the folders before saving the dataset
    os.makedirs("train_dataset", exist_ok=True)
    train_dataset.save_to_disk("./train_dataset")
    os.makedirs("val_dataset", exist_ok=True)
    val_dataset.save_to_disk("./val_dataset")
    # os.makedirs("test_dataset", exist_ok=True)
    # test_dataset.save_to_disk("./test_dataset")

#creater the dataloader for the train and test dataset with each having same length using the collate_fn
def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_values = [torch.tensor(i).squeeze() for i in input_values]
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True,padding_value=641)
    return {"input_values":input_values,"labels":torch.tensor(labels).squeeze()}
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=config.batch_size, 
                                           shuffle=True,collate_fn=collate_fn,drop_last=True,worker_init_fn=seed_worker,generator=g,)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False,collate_fn=collate_fn,drop_last=True,worker_init_fn=seed_worker,generator=g,)


# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), num_classes)
        
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.gru(x, h0)  
        # or:
        # out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
        out = torch.nn.functional.relu(out)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        # out: (n, 10)
        return out
from accelerate import Accelerator
accelerator = Accelerator()

# model = RNN(config.input_size, config.hidden_size, config.num_layers, num_classes).to(device)
ntokens = 6 # the size of vocabulary
emsize = 642 # embedding dimension
config.nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
config.nlayers = 10 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
config.nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
encoding_size = 642
model = TransformerModel(encoding_size,ntokens, emsize, config.nhead, config.nhid, config.nlayers, dropout).to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
model, optimizer, train_loader  = accelerator.prepare(
    model, optimizer, train_loader
)
best_acc = 0
lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(config.num_epochs*(len(train_loader))*0.1),
        num_training_steps=config.num_epochs*(len(train_loader)),
    )
# Train the model
n_total_steps = len(train_loader)
for epoch in range(config.num_epochs):
    loss_epoch =0
    for i, values in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        images = values["input_values"].to(device,dtype=torch.long)

        # images = values["input_values"].to(device)
        labels= values["labels"]
        labels = labels.type(torch.LongTensor).to(device)
        # Forward pass
    
        outputs = model(images,None)
        loss = criterion(outputs.mean(1),labels )       
        # Backward and optimize
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        
        loss_epoch += accelerator.gather(loss)
        if accelerator.is_main_process:
            if (i+1) % 1000 == 0:
                print (f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    if accelerator.is_main_process:
        wandb.log({"loss": loss_epoch.sum()/(len(train_dataset)/config.batch_size)})
        print("loss per epoch",loss_epoch.sum()/(len(train_dataset) / config.batch_size))
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            val_loss = 0
            for i, values in enumerate(val_loader):
                # images = torch.nn.functional.one_hot(values["input_values"], num_classes=642).to(device)
                images = values["input_values"].to(device,dtype=torch.long)
                labels = values["labels"]
                labels = labels.type(torch.LongTensor).to(device)
                outputs = model(images,None)
                loss = criterion(outputs.mean(1),labels )
                val_loss += loss.item()
                # max returns (value ,index)
                _, predicted = torch.max(outputs.mean(1).data, 1)
                n_correct += (predicted == labels).sum()
            acc = (100.0 * n_correct )/ (len(val_loader)*config.batch_size)
            wandb.log({"validation accuracy": acc})
            wandb.log({"validation loss": val_loss/(len(val_loader))})
            print(f'Accuracy of the network on the test images: {acc} %')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_model_transformer.pt')
                print('model saved')
                print(f'Best Accuracy of the network on the test images: {best_acc} %')