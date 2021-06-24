# monk-text-simplification

本项目是窦帅（北京大学信息科学技术学院研究生）对于文本简化任务自主设计的paraphrase+classification模型项目源码，文本简化的输入是一个复杂句子，然后通过对于进行复杂单词的替换，删除，改写等操作，输出一个简单句子。

### motivation

之前的任务主要采用的方式是seq2seq的有监督学习,但是这种有监督的训练有一个缺点那就是数据集的质量不高。文本简化训练集是wikipedia和简单wiki-pedia。简单wikipedia是Google专门针对青少年出的对wikipedia的简化版本。但是因为二者之间句子的对齐是机器自动对齐的，所以会导致很多句子对不是互相对应的关系。数据集本身质量很差。有监督模型训练出来的效果也就不是很理想,所以我们希望通过一种无监督的方式来实现文本的简化。

## model

为此我们把简化任务看成一个改写加分类任务。训练一个改写模型，对于一个输入的复杂句子，输出多个改写版本的句子。然后训练一个句子难度复杂度的分类器用来对句子的难度进行评定，之后将之前生成的多个改写句子输入到分类器中，最终输出一个最简单的句子。

我们的模型在输入端，把句子输入t5-paraphrase模型中输出多个候选句子，最终用一个bert难度判别器来输出最简单的句子作为我们的输出。但是这样模型遇到一个问题，因为是paraphrase模型，所以模型输出句子的长度和原始复杂句子长度基本一致，而reference的句子长度普遍小于原始复杂句子，因此我们在输入前进行一个BART句子压缩预处理，对于长句子能够首先进行一个句子压缩的预处理，然后输入到我们模型中去。

## result

|      |                           | SARI-turk | FKGL-turk | SARI-asset | FKGL-asset |
| ---- | ------------------------- | --------- | --------- | ---------- | ---------- |
| SOTA | MUSS                      | 40.85     | 8.79      | 42.65      | 8.23       |
| monk | paraphrase+classification | 39.606    | 9.56      | 42.7       | 6.523      |

注：SARI越大越好，FKGL越小越好，turk代表turkcorpus数据集，asset代表asset数据集。

通过实验进行对比，我们的模型在turkcorpus数据集上

## Controable sentence simplification

之前我们的模型可以实现对一个复杂句子生成一个简单句子，但是我们进一步进行思考，希望实现的对一个句子，生成不同难度等级的句子。这样就可以实现之前提到的在到教育领域，对一个句子，给不同等级的学习者输出适合他们的不同等级的简化句子。我们把这个任务定义为可控文本简化。我们认为可控文本简化属于可控文本生成的一个子问题。因此我们通过调研，决定采取2020ICLR一篇PPLM（https://arxiv.org/abs/1912.02164）的可控文本生成的论文作为我们的方案，原因是这个PPLM是plug and play,像图像中的老鼠一样可以和大象一般的预训练模型做即插即用似得绑定。他们处理的问题是生成不同情感 negative或者 positive的句子，这和我们句子难度simple，hard很类似。它会首先训练一个attribute model，这个分类器是判定你是negative还是positive的情感倾向，之后在decoder的时候每一个step的hidden state都会输入到这个attribute model中来对其进行动态调整。从而改变输出单词的分布，让其倾向于输出对应情感的句子。

**我们对这个模型的主要更改是，原始模型的代码是针对GPT这种只有decoder的预训练模型，而我们采用的是t5这种encoder-decoder的模型架构，所以在基本思想不变的前提下，需要对其代码以及逻辑进行一些更改和优化，相关代码工作已经完成，目前正处于调优和实验阶段。**

## code

t5train文件夹为训练t5模型，需要新创建数据文件夹wiki-auto。wiki-auto数据来源：https://github.com/chaojiang06/wiki-auto

Bertclaasification文件夹为训练Bert难度评定器，需要创建数据文件夹wikiData. 我们默认wikilarge的source为复杂句子，target为简单句子。数据来源：https://www.aclweb.org/anthology/D17-1062/

bartTrain为训练bart文本压缩模型，所用的是fairseq训练，直接运行train_bart.sh,用bart生成则运行generate_bart.py脚本

evaluate是将生成模型进行服务封装，实现随意给定一个句子，用模型生成简化句子。

PPLM文件夹为对原始PPLM代码进行更改，t5discrim_v1.1.py 代表训练复杂简单句子的discriminator，run_pplm.py 代表运行pplm生成句子。



