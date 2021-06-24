hardFile = open("./wikiData/wikilarge/wiki.full.aner.train.src","r")
hardLines = hardFile.readlines()
hardLabel = [1 for i in range(len(hardLines))]
easyFile = open("./wikiData/wikilarge/wiki.full.aner.train.dst","r")
easyLines = easyFile.readlines()
easyLabel = [0 for i in range(len(easyLines))]

train_texts, train_labels = hardLines+easyLines,hardLabel+easyLabel
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# train_texts = train_texts[:1000]
# val_texts = val_texts[:1000]
# train_labels = train_labels[:1000]
# val_labels = val_labels[:1000]

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
val_dataset = Dataset(val_encodings, val_labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=6,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    evaluate_during_training=True,
    logging_steps=1000,
    save_steps=10000
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()w