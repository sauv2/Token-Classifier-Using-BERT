from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

def random_set(val,array,array2):
    
    random_indices=random.sample(range(len(array)),val)
    new_arr=[]
    new_arr1=[]

    for i in random_indices:
        new_arr.append(array[i])
        new_arr1.append(array2[i])
    return new_arr,new_arr1

def compute_metrics(preds,labels):

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="weighted",zero_division=0)
    recall = recall_score(labels, preds, average="weighted",zero_division=0)
    f1 = f1_score(labels, preds, average="weighted",zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def helper(data):
    input=tokenizer(data["tokens"],truncation=True, is_split_into_words=True)

    label_final=[]
    for i,label in enumerate(data[f"ner_tags"]):
        id=input.word_ids(batch_index=i)
        prev=None
        id_append=[]
        for idx in id:
            if idx is None:
                id_append.append(-100)
            elif idx!=prev:
                id_append.append(label[idx])
            else:
                id_append.append(-100)
                prev=idx
        label_final.append(id_append)
    input["labels"]=label_final
    return input

