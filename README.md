# CT-Transformer


## 简介
Controllable Time-delay Transformer是达摩院语音团队提出的语音识别后处理框架中的标点恢复和顺滑检测模块。本项目尝试复现该模型，可以被应用于文本类输入的标点预测，也可应用于语音识别结果的顺滑检测，协助语音识别模块输出具有可读性的文本结果。

示例：“I want a flight to Boston um to Denver”

对于标点预测，我们需要预测每个词后是否有某种标点，比如逗号、问号、句号。在这个例子中，Denver这个词后会有一个句号。对于顺滑检测，主要包括检测三种需要被顺滑的类型，RM（Reparandum）、RP（Repair）和IM（Interregnum）。RM是被后面纠正或者被丢弃的词组，包括重复词，被修复词等。IM则指停顿词、语气词等。对于前面的例子，词组“to Boston”需要被检测为RM，词组“to Denver”需要被检测为RP，单词“um”需要被检测为IM。经过标点预测和去除检测出来的RM和IM词组，并保留RP词组，原来的ASR 输出文本转化为 “I want a flight to Denver.”


## 模型结构

定义在[model.py](model.py)，更多细节可见原论文[Controllable Time-Delay Transformer for Real-Time Punctuation Prediction and Disfluency Detection](https://arxiv.org/pdf/2003.01309)

## 预训练

预训练任务同时做了标点恢复和顺滑检测任务，使用了baikeqa和wiki_zh数据（https://github.com/brightmart/nlp_chinese_corpus），生成数据的代码可见[create_dataset.py](create_dataset.py)，预训练代码可见[pretrain.py](pretrain.py)


## 微调

顺滑检测使用csc数据https://github.com/shibing624/pycorrector/tree/master/examples/data/sighan_2015，生成数据的脚本可见[create_dataset.py](create_dataset.py)，微调代码[finetune.py](finetune.py)

标点恢复任务使用iwslt2012_zh数据https://github.com/jiangnanboy/punctuation_prediction/blob/main/data/iwslt2012_zh.rar。


## 效果

```
Input: 因为我们也做商标商标申请，商标标呃专利申请啊，所以这这个这特定的专业组开始可能需要需要特定的这个写呃那个一些数据库和系统啊，那这些可能都是软件啊。

Output: 因为我们也做商标申请，商标专利申请，所以这个特定的专业组开始可能需要特定的这个写那个一些数据库和系统，那这些可能都是软件啊。 

运行时间(CPU): 17.20

Input: 当当然可能可能刚好就那个呃，他那边对也也也也是我的客户，那我不能代表您去告他，对不对？那对对对对，这就有一冲了，这才是这个概念。

Output: 当然可能刚好就那个，他那边对也是我的客户，那我不能代表您去告他，对不对？那对，这就有一冲了，这才是这个概念。

运行时间(CPU): 7.93 毫秒

```
