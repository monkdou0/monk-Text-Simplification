from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
from FKGL import fkgl
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
    encoding = tokenizer_t5.encode_plus(text, pad_to_max_length=True, return_tensors="pt").to(torch_device)
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

def getParaphrasePegasus(sentence,decode="beam"):

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer_t5([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        # batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt")
        # translated = model_t5.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        translated = model_t5.generate(**batch,max_length=256,num_beams=num_beams, num_return_sequences=num_return_sequences, no_repeat_ngram_size=2,early_stopping=True)

        tgt_text = tokenizer_t5.batch_decode(translated, skip_special_tokens=True)
        return tgt_text
    def get_response_sample(input_text,num_return_sequences):
        batch = tokenizer_t5([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        # batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt")
        # translated = model_t5.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        translated = model_t5.generate(**batch,max_length=256,do_sample=True,top_k=200,top_p=0.95,early_stopping=True, num_return_sequences=10)
        tgt_text = tokenizer_t5.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    if decode=="beam":res = get_response(sentence,10,10)
    elif decode=="sample":res = get_response_sample(sentence,10)

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


def getBestFKGLsentence(sentences):
    threshold = 10000000
    simpleSen = ""
    for sen in sentences:
        score = fkgl.getFKGL(sen)
        if score<threshold:
            threshold = score
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




import argparse
def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="tuner007/pegasus_paraphrase", help="model name")
    parser.add_argument('--tokenizer', type=str, default="tuner007/pegasus_paraphrase", help="tokenizer name")
    # parser.add_argument('--filename', type=str, default="result-pegasus-new", help="tokenizer name")
    parser.add_argument('--input', type=str, default="./output/asset.test.orig", help="input name")
    return parser.parse_args()

if __name__ =="__main__":


    args =ArgumentParser()
    model_t5 = args.model
    tokenizer_t5 = args.tokenizer
    # filename = args.filename
    inputFile = args.input
    inputFile = "./output/asset.test.orig"


    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tokenizer_t5 = AutoTokenizer.from_pretrained(tokenizer_t5)
    # model_t5 = AutoModelForSeq2SeqLM.from_pretrained(model_t5).to(torch_device)

    # model_name = 'tuner007/pegasus_paraphrase'

    tokenizer_t5 = PegasusTokenizer.from_pretrained(model_t5)
    model_t5 = PegasusForConditionalGeneration.from_pretrained(model_t5).to(torch_device)



    f = open(inputFile,"r")
    turkcorpus = f.readlines()

    turkcorpus = [i.strip() for i in turkcorpus]

    # nums = np.random.randint(1 ,1111111,15)
    # nums = ["1035252","1695","5678"]
    nums = [1035252]
    for num in nums:
        # filename = "./pegasusRes/pegasus-fkglClassification-sampleDecoder-random-"+str(num)
        filename = "./res.txt"
        setup_seed(int(num))
        wf = open(filename,"w")
        i = 0
        for sentence in turkcorpus:
            score,ssen = getBestFKGLsentence(getParaphrasePegasus(sentence,decode="sample"))
            print(ssen,file=wf)
        wf.close()
