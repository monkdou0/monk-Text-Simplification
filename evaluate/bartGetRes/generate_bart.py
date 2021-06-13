"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel
import os

model_path = "paw_ckpt_bart"
data_name_or_path = "checkpoint_best.pt"
bart = BARTModel.from_pretrained('./bartGetRes/paw_ckpt_bart/', checkpoint_file='checkpoint_best.pt')
bart.eval()

def bartGenerate(sen):
    preds = bart.sample([sen], beam=4, lenpen=2.0, no_repeat_ngram_size=2, temperature=0.9)
    return preds[0]


    # with open('asset2.orig',"r") as source:
    #     lines = [line.replace("\n", "") for line in source.readlines()]
    #     with torch.no_grad():
    #         preds = bart.sample(lines, beam=4, lenpen=2.0, no_repeat_ngram_size=2, temperature=0.9)
    #         for i, (line, pred) in enumerate(zip(lines, preds)):
    #             print(pred)
    # preds = bart.sample(lines, beam=4, lenpen=2.0, no_repeat_ngram_size=2, temperature=0.9)
    # for i, (line, pred) in enumerate(zip(lines, preds)):
    #     print(pred)




if __name__ == "__main__":
    sen = "The spacecraft consists of two main elements: the NASA Cassini orbiter, named after the Italian-French astronomer Giovanni Domenico Cassini, and the ESA Huygens probe, named after the Dutch astronomer, mathematician and physicist Christiaan Huygens."
    print(bartGenerate((sen)))
