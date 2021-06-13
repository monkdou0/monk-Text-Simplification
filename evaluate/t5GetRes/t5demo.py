from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import torch
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(1019)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('./t5GetRes/model')


monk_model = "./t5GetRes/monkmodelSumm"
model_vamsi = "Vamsi/T5_Paraphrase_Paws"
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(monk_model)


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


def t5Generate(sentence):
    score,ssen = getBestSimSentence(getParaSentences(sentence))
    return ssen




if __name__=="__main__":
    sentence = "this is the best time to see what you can do, and this is the best time to do what you can do."
    sim = t5Generate(sentence)
    print(type(sim))
    print(sim)