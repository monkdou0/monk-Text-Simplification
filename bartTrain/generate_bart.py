"""Generation using BART"""
import torch
from fairseq.models.bart import BARTModel
import os
def getFiles(folder):

    path = folder
    files= os.listdir(path) #得到文件夹下的所有文件名称
    s = []
    for file in files: #遍历文件夹
        if not os.path.isdir(path+"/"+file): #判断是否是文件夹，不是文件夹才打开
            s.append(path+"/"+file)
    return s


def main():

    bart = BARTModel.from_pretrained('ckpt_bart', checkpoint_file='checkpoint_best.pt')
    #bart.cuda()
    #bart.half()
    bart.eval()

    pegasusFiles = getFiles("./shuai/pegasus")
    for pfile in pegasusFiles:
        with open(pfile,"r") as source:
            lines = [line.replace("\n", "") for line in source.readlines()]
            file = open(pfile+"-bartPostprocess","w")

            with torch.no_grad():
                preds = bart.sample(lines, beam=4, lenpen=2.0, no_repeat_ngram_size=2, temperature=0.9)
                for i, (line, pred) in enumerate(zip(lines, preds)):
                    print(pred,file=file)
            file.close()




if __name__ == "__main__":
    main()
