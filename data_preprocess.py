from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForTokenClassification, AdamW
from transformers import BertConfig
from transformers import DataCollatorForTokenClassification
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
import evaluate
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


import tokenizer_code as tc
from tokenizer_code import compute_metrics as cm

# Retrieving data
data = load_dataset("wnut_17")

# Tokenizing the data, as data tokenizing does not match our parameters

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_output = data.map(tc.helper, batched=True)

print(tokenized_output["train"][0]["labels"])
print(tokenized_output["train"][0]["input_ids"])
# Converting the data into tensors and aligning it
padded_input_ids = pad_sequence([torch.tensor(seq) for seq in tokenized_output["train"]["input_ids"]], batch_first=True, padding_value=tokenizer.pad_token_id)
padded_labels = pad_sequence([torch.tensor(seq) for seq in tokenized_output["train"]["labels"]], batch_first=True, padding_value=tokenizer.pad_token_id)

#for smaller batch

#padded_input_ids,padded_labels=tc.random_set(1024,padded_input_ids,padded_labels)

#padded_input_ids_tensor = torch.stack(padded_input_ids)
#padded_labels_tensor = torch.stack(padded_labels)

train_dataset = TensorDataset(padded_input_ids, padded_labels)

# Defining the parameters for the model

num = 13
id_label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
}

label_id  = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

lr = 1e-5
train_batch_size = 12
eval_size = 12
epoch_sz = 2
w_dec = 0.01
eval_strat = "epoch"
save_strat = "epoch"
H=1152
H_L=18
max_norm=1.0

# Defining the model and optimizer
num_labels = 13


#modifying the hidden layers and hidden layers size

config = BertConfig.from_pretrained("bert-base-uncased", num_labels=13, id2label=id_label, label2id=label_id)
config.num_hidden_layers=6
config.hidden_size=384

#structure of the model

model = BertForTokenClassification(config)
#freezing
#for param in model.bert.embeddings.parameters():
    #param.requires_grad = False

#for param in model.bert.encoder.layer[:3].parameters():
        #param.requires_grad = False


optimizer = AdamW(model.parameters(), lr=lr, weight_decay=w_dec)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
#training loop


for epoch in range(epoch_sz):
    #writing down parameter constants
    loss_avg=0 
    precision=0 
    f1=0 
    recall=0 
    accuracy=0
    num=0
    max_norm=1.0
    
    #starting training

    model.train()
    
    for batch in DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True):
        #check target size matches the desired size, else it will throw an error
        inputs, targets = batch
        assert targets.max() < num_labels
        #forward pass
        outputs = model(input_ids=inputs, labels=targets)
        

        optimizer.zero_grad()
            
        loss = loss_fn(outputs.logits.view(-1,num_labels), targets.view(-1))
        loss_avg+=loss
        

        #preparing parameters for computing metrics
        pred=outputs.logits.argmax(dim=-1).view(-1)
        label=targets.view(-1)
        

        #computing loss, precision, f1 score, accuracy
        num+=1
        
        precision+=tc.compute_metrics(preds=pred,labels=label)["precision"]
        accuracy+=tc.compute_metrics(preds=pred,labels=label)["accuracy"]
        f1+=tc.compute_metrics(preds=pred,labels=label)["f1"]
        recall+=tc.compute_metrics(preds=pred,labels=label)["recall"]
        
        
        print(f"itr: {num}")
        print(f"p:{precision/num}, a:{accuracy/num}")

        #backward pass
        loss.backward()
        #gradient clipping
        
        clip_grad_norm_(model.parameters(),max_norm)
        optimizer.step()
        
        #Learning rate
        scheduler.step()


    loss_avg=loss_avg/num
    L=f"For Epoch {epoch+1}, Loss Avg = {loss_avg}, Precision={precision/num}, Accuracy={accuracy/num}, f1 score={f1/num}, recall={recall/num}, learning rate={lr}"
    print(L)
    file1=open("records.txt", "a")
    file1.write(f"For Epoch {epoch+1}, Loss Avg = {loss_avg}, Precision={precision/num}, Accuracy={accuracy/num}, f1 score={f1/num}, recall={recall/num}, learning rate={lr} , Hidden Layer Size={H}, Hidden Layers={H_L}\n")
    file1.close()

torch.save(model,'weights.pth')