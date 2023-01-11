import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from transformers import get_scheduler
import os
import random
import seaborn as sns
import numpy as np
from cp_feature_extractor import model_output
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
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
wandb.init(project='lstm_langmodel_batch')
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 



config = wandb.config
config.batch_size = 8
config.num_epochs = 500
# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 1e-3
config.one_hot = True
config.hidden_size = 1024
config.num_layers  = 2
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
if os.path.exists("train_dataset_wav2vec2")and os.path.exists("val_dataset_wav2vec2") :
    train_dataset = Dataset.load_from_disk("train_dataset_wav2vec2")
    val_dataset = Dataset.load_from_disk("val_dataset_wav2vec2")
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
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True,padding_value=640)
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
    def __init__(self,  hidden_size, num_layers, num_classes,ntoken, ninp):
        super(RNN, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=ntoken-1)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        self.gru = nn.GRU(ninp, hidden_size, num_layers, batch_first=True,bidirectional=True)
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size*2, int(hidden_size))
        self.fc2 = nn.Linear(int(hidden_size), num_classes)
        
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        x = self.encoder(x)
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) 
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
print(torch.cuda.device_count())
print(accelerator.state.num_processes)
config.input_size = 641
ninp = 512
model = RNN( config.hidden_size, config.num_layers, num_classes,config.input_size, ninp).to(device)

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
def calculate_class_prb_sum(prediced_prb,labels,class_prob_sum):
    # add all the corresponding indices of prediced_prb depending on the labels
    prediced_prb = prediced_prb.to("cpu")
    labels = labels.to("cpu")
    for i, label in enumerate(labels):
        if int(label) not in class_prob_sum:
            class_prob_sum[int(label)] = prediced_prb[i]
        class_prob_sum[int(label)] = list(map(lambda x, y: x + y, class_prob_sum[int(label)], prediced_prb[i]))
    return class_prob_sum
# Train the model
n_total_steps = len(train_loader)
for epoch in range(config.num_epochs):
    loss_epoch =0
    for i, values in enumerate(train_loader):  
        # origin shape: [N, 1, 28, 28]
        # images = torch.nn.functional.one_hot(values["input_values"], num_classes=641).to(device)
        #set images dtype to float32
        # images = values["input_values"].type(torch.float32)
        images = values["input_values"].to(device,dtype=torch.long)
        # images = values["input_values"].to(device)
        labels= values["labels"]
        labels = labels.type(torch.LongTensor).to(device)
        # Forward pass

        outputs = model(images)
        loss = criterion(outputs,labels )       
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
        wandb.log({"loss": loss_epoch.sum()/(len(loss_epoch)*len(train_loader))})
        print("loss per epoch",loss_epoch.sum()/(len(loss_epoch)*len(train_loader)))
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            val_loss = 0
            prediced_list = []
            labels_list=[]
            prediced_prb_list = []
            class_prob_sum = {}
            for i, values in enumerate(val_loader):
                # images = torch.nn.functional.one_hot(values["input_values"], num_classes=641).to(device)
                # images = images.type(torch.float32)
                images = values["input_values"].to(device,dtype=torch.long)
                labels = values["labels"]
                labels = labels.type(torch.LongTensor).to(device)
                outputs = model(images)
                loss = criterion(outputs,labels )
                val_loss += loss.item()
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
               
                prediced_prb = torch.nn.functional.softmax(outputs.data, dim=1)

                prediced_list += predicted.tolist()
                n_correct += (predicted == labels).sum()
                class_prob_sum = calculate_class_prb_sum(prediced_prb,labels,class_prob_sum)
                labels_list += labels.tolist()
            acc = (100.0 * n_correct )/ (len(val_loader)*config.batch_size)
            print(len(prediced_list))
            print(len(val_dataset['labels']))
            f1_score_val = f1_score(labels_list, prediced_list, average='weighted')
                
            wandb.log({"validation accuracy": acc})
            wandb.log({"validation f1_score": f1_score_val})
            wandb.log({"validation loss": val_loss/(len(val_loader))})
            print(f'Accuracy of the network on the test images: {acc} %')
            print(f'f1_score of the network on the test images: {f1_score_val} %')
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), f"best_model_lstm_batch{model_checkpoint.split('/')[-1]}_bidirectional.pt")
                print('model saved')
                print(f'Best Accuracy of the network on the test images: {best_acc} %')
                cf_m = confusion_matrix(labels_list, prediced_list)
                plt.figure(figsize=(10,10))
                sns.heatmap(cf_m, annot=True, fmt="d",xticklabels=lang, yticklabels=lang)
                plt.title("Confusion matrix")
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.show()
                #save the plot as png
                plt.savefig(f"confusion_matrix_{model_checkpoint.split('/')[-1]}_bidirectional.png")
                plt.figure(figsize=(10,10))
                class_prob = {}
                for k,v in class_prob_sum.items():
                    lst = np.array((torch.tensor(v)/torch.tensor(v).sum()).tolist())
                    #for each element in "a" round it to 3 decimal points and removing the extra trailing zeroes
                    tensor_list = [round(elem, 3) for elem in lst]
                    # tensor_list =  np.around(np.array((torch.tensor(v)/torch.tensor(v).sum()).tolist()), decimals=3).reshape(-1)
                
                    class_prob = dict(zip(class_prob_sum.keys(),tensor_list))
                    class_prob_sum[k] = class_prob
                sns.heatmap(pd.DataFrame(class_prob_sum), annot=True, fmt="f",xticklabels=lang, yticklabels=lang)
                plt.title("Confusion matrix probability")
                plt.ylabel('True label') 
                plt.xlabel('Predicted label')
                plt.show()
                #save the plot as png
                plt.savefig(f"confusion_matrix_prob_{model_checkpoint.split('/')[-1]}_bidirectional.png")