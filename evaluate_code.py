from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from transformers import BertConfig
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_

import tokenizer_code as tc
from tokenizer_code import compute_metrics as cm

# Retrieving data
data = load_dataset("wnut_17")

# Tokenizing the data, as data tokenizing does not match our parameters

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_output = data.map(tc.helper, batched=True)

# Converting the data into tensors and aligning it
padded_input_ids = pad_sequence([torch.tensor(seq) for seq in tokenized_output["test"]["input_ids"]], batch_first=True, padding_value=tokenizer.pad_token_id)
padded_labels = pad_sequence([torch.tensor(seq) for seq in tokenized_output["test"]["labels"]], batch_first=True, padding_value=tokenizer.pad_token_id)




test_dataset = TensorDataset(padded_input_ids, padded_labels)

model=BertForTokenClassification.from_pretrained("bert-base-uncased")
loaded_model = torch.load('weight.pth', map_location=torch.device('cpu'))

batch_sz=12

test_data = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)

model.eval()
model.config.num_labels=13
import torch

loss_fn = torch.nn.CrossEntropyLoss()
num = 0
loss_avg = 0.0
precision = 0.0
accuracy = 0.0
f1 = 0.0
recall = 0.0

with torch.no_grad():
    for batch in test_data:
        num += 1
        inputs, targets = batch
        

        attention_mask = (inputs != tokenizer.pad_token_id).float()
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
       
        

        pred = outputs.logits.argmax(dim=-1).view(-1)
        label = targets.view(-1)

        precision += tc.compute_metrics(preds=pred, labels=label)["precision"]
        accuracy += tc.compute_metrics(preds=pred, labels=label)["accuracy"]
        f1 += tc.compute_metrics(preds=pred, labels=label)["f1"]
        recall += tc.compute_metrics(preds=pred, labels=label)["recall"]

        print(f"Iteration: {num}")
        print(f"Precision: {precision / num}, Accuracy: {accuracy / num}")
        
loss_avg /= num
precision /= num
accuracy /= num
f1 /= num
recall /= num

print(f"The evaluation metrics are: Precision = {precision}, Accuracy = {accuracy}, F1 score = {f1}, and Recall = {recall}")

