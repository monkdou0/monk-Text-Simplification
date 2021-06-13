from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

import numpy as np

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('./model')

def getHardProb(sentence):
    try:
        inputs = tokenizer(sentence,return_tensors="pt")
        outputs = model(**inputs)
        output = outputs[0][0]
        output = np.exp(output.detach().numpy())
        prob = output/sum(output)

        return prob[1]
    except:
        return 0


def getParaSentences(sentence):
    res = []
    text = "paraphrase: " + sentence
    encoding = tokenizer_t5.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
    outputs = model_t5.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=10)
    for output in outputs:
        line = tokenizer_t5.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        res.append(line)
    return res


def getBestSimSentence(sentences):
    threshold = 1
    simpleSen = ""
    for sen in sentences:
        hardScore = getHardProb(sen)
        if hardScore<threshold:
            threshold = hardScore
            simpleSen = sen
    return threshold,simpleSen


import torch
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(1019)


import argparse
def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Vamsi/T5_Paraphrase_Paws", help="model name")
    parser.add_argument('--tokenizer', type=str, default="t5-base", help="tokenizer name")
    parser.add_argument('--filename', type=str, default="result-20210114", help="tokenizer name")
    parser.add_argument('--input', type=str, default="asset.orig", help="input name")
    return parser.parse_args()

if __name__ =="__main__":


    args =ArgumentParser()
    model = args.model
    tokenizer = args.tokenizer
    filename = args.filename
    inputFile = args.input


    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_t5 = AutoTokenizer.from_pretrained(tokenizer)
    model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model)


    f = open(inputFile,"r")
    turkcorpus = f.readlines()

    turkcorpus = [i.strip() for i in turkcorpus]


    wf = open(filename,"w")
    i = 0
    for sentence in turkcorpus:
        score,ssen = getBestSimSentence(getParaSentences(sentence))
        print(ssen,file=wf)
        i = i + 1
        print(i)
