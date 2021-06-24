# Importing stock libraries
import numpy as np
import pandas as pd
import os
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# WandB – Import the wandb library
import wandb


# # Setting up the device for GPU usage
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
MODELLOCATION = "./wiki-auto/model/wiki-auto-model"
WANDBNAME = "wiki-auto"
TRAINLOCATION = 'wiki-auto/wiki_auto_train.csv'
TESTLOCATION = 'wiki-auto/wiki_auto_test.csv'
# TRAINLOCATION = "./paradata/PAW_Train.csv"

# Creating a custom dataset for reading the dataframe and loading it into the dataloader to pass it to the neural network at a later stage for finetuning the model and to prepare it for predictions

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, pad_to_max_length=True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.summ_len, pad_to_max_length=True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


# Creating the training function. This will be called in the main function. It is run depending on the epoch value.
# The model is put into train mode and then we wnumerate over the training loader and passed to the defined network

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    lossHistory = float("inf")
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]
        loss = torch.sum(loss)
        print("--------------------------------------------------------")
        print(loss)
        print(type(loss))
        print("--------------------------------------------------------")
        if _ % 10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # xm.optimizer_step(optimizer)
        # xm.mark_step()

        loss = loss.item()
        if loss<lossHistory:
            lossHistory = loss
            model.module.save_pretrained(MODELLOCATION)

def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


def main():
    # WandB – Initialize a new run
    wandb.init(project=WANDBNAME)

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = wandb.config  # Initialize config
    config.TRAIN_BATCH_SIZE = 4  # input batch size for training (default: 64)
    config.VALID_BATCH_SIZE = 4  # input batch size for testing (default: 1000)
    config.TRAIN_EPOCHS = 2  # number of epochs to train (default: 10)
    config.VAL_EPOCHS = 1
    config.LEARNING_RATE = 1e-4  # learning rate (default: 0.01)
    config.SEED = 42  # random seed (default: 42)
    config.MAX_LEN = 512
    config.SUMMARY_LEN = 150

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # Importing and Pre-Processing the domain data
    # Selecting the needed columns only.
    # Adding the summarzie text in front of the text. This is to format the dataset similar to how T5 model was trained for summarization task.

    # summarization
    # df = pd.read_csv('./data/news_summary.csv', encoding='latin-1')
    # df = df[['text', 'ctext']]
    # df.ctext = 'summarize: ' + df.ctext
    # print(df.head())

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest will be used for validation.
    # train_size = 0.8
    # train_dataset = df.sample(frac=train_size, random_state=config.SEED)
    # val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    # train_dataset = train_dataset.reset_index(drop=True)

    train_dataset = pd.read_csv(TRAINLOCATION, header=None)
    train_dataset.columns = ['ctext', 'text']
    train_dataset = train_dataset.reset_index(drop=True)

    val_dataset = pd.read_csv(TESTLOCATION, header=None)
    val_dataset.columns = ['ctext', 'text']
    val_dataset = val_dataset.reset_index(drop=True)

    train_dataset.ctext = 'paraphrase: ' + train_dataset.ctext
    val_dataset.ctext = 'paraphrase: ' + val_dataset.ctext

    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(val_dataset.shape))

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = CustomDataset(val_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = nn.DataParallel(model)
    model = model.to(device)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(config.TRAIN_EPOCHS):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    # Validation loop and saving the resulting file with predictions and acutals in a dataframe.
    # Saving the dataframe as predictions.csv
    # print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')
    # model = T5ForConditionalGeneration.from_pretrained(MODELLOCATION)
    # model = model.to(device)
    # for epoch in range(config.VAL_EPOCHS):
    #     predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
    #     final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    #     final_df.to_csv('./models/paw-t5large.csv')
    #     print('Output Files generated for review')


if __name__ == '__main__':
    main()
