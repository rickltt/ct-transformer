import json
import os
import glob
import time
import torch
from tqdm import tqdm
from seqeval.metrics import accuracy_score, f1_score, classification_report

from config import CTConfig
from tokenizer import CTTokenizer
from model import CTTransformerForDisfluDetection

config = CTConfig("./config.json")
tokenizer = CTTokenizer("./vocab.json")

model_path = "./output/finetune_output_disflu/last_checkpoint.pt"

disflu_labels = ["O", "B-IM", "I-IM", "B-RM", "I-RM", "B-RP", "I-RP"]
# punct_labels = ["O",'，','。', '？']

num_disflu_labels = len(disflu_labels)
# num_punct_labels = len(punct_labels)

config.num_disflu_labels = num_disflu_labels
# config.num_punct_labels = num_punct_labels
    
model = CTTransformerForDisfluDetection(config)
state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(state_dict)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
model.eval()

# class CTInfer:
#     def __init__(self, model, tokenizer, max_length=256, frame_rate=10):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.frame_rate = frame_rate
#         # self.look_ahead = look_ahead
#         self.max_length = max_length
#         self.buffer = []

#     def predict(self, tokens):
#         inputs = tokenizer(tokens, is_split_into_words=True, max_length=self.max_length, return_pt=True)

#         output = model(**inputs)
#         disflu_logits, punct_logits = output[1:]

#         disflu_preds = disflu_logits.argmax(-1).view(-1).tolist()
#         punct_preds = punct_logits.argmax(-1).view(-1).tolist()

#         disflu_preds = disflu_preds[1:len(tokens)+1]
#         punct_preds = punct_preds[1:len(tokens)+1]

#         return disflu_preds, punct_preds
    
#     def infer(self, text):
#         tokens = tokenizer.tokenize(text)
#         tokens = tokens[:self.max_length - 2] if len(tokens) > self.max_length - 2 else tokens
#         start_time = time.perf_counter()
#         result = []
#         for i in range(0, len(tokens), self.frame_rate):
#             new_words = tokens[i:i+self.frame_rate]
#             self.buffer.extend(new_words)
            
#             disflu_preds, punct_preds = self.predict(self.buffer)
            
#             # self.disflu_buffer.extend(disflu_preds)
#             # self.punct_buffer.extend(punct_preds)
#             assert len(disflu_preds) == len(punct_preds) == len(self.buffer)
#             print(self.buffer)
#             print(punct_preds)
#             print(disflu_preds)
#             print("result:", result)
#             for i, pred in enumerate(punct_preds):
#                 if punct_labels[pred] in ["。","？"]:  # 假设'.'和'?'是句子结束符
                    
#                     pop_tokens = self.buffer[:i+1]
                    
#                     self.buffer = self.buffer[i + 1:]
#                     # if len(self.buffer) >= self.look_ahead:
#                     #     break
#                     disflu_preds = disflu_preds[:i+1]
#                     punct_preds =  punct_preds[:i+1]
                    
                    
#                     assert len(pop_tokens) == len(disflu_preds) == len(punct_preds)

#                     tmp_result = []
#                     for token, disflu_pred, punct_pred in zip(pop_tokens, disflu_preds, punct_preds):
#                         if disflu_pred in [0,5,6]:
#                             tmp_result.append(token)
#                         if punct_pred != 0:
#                             tmp_result.append(punct_labels[punct_pred])
#                     result.extend(tmp_result) 
#                     break

#         for token, disflu_pred, punct_pred in zip(self.buffer, disflu_preds, punct_preds):
#             if disflu_pred in [0,5,6]:
#                 result.append(token)
#             if punct_pred != 0:
#                 result.append(punct_labels[punct_pred])
                
#         final_result = "".join(result)
#         self.buffer = []
#         end_time = time.perf_counter()
#         elapsed_time_ms = (end_time - start_time) * 1000
#         print(f"运行时间: {elapsed_time_ms:.2f} 毫秒")
        
#         return final_result
        

def predict(text):

    start_time = time.perf_counter()
    # max_length=256
    # text = "啊嗯对哦，有蓝色有蓝的那个哦，有logo是吧？Logo对花花式咖啡啡"
    tokens = tokenizer.tokenize(text)
    print(tokens)
    # tokens = tokens[:max_length - 2] if len(tokens) > max_length - 2 else tokens
    inputs = tokenizer(text, return_pt=True)
    output = model(**inputs)
    disflu_logits = output[1]

    disflu_preds = disflu_logits.argmax(-1).view(-1).tolist()
    print(disflu_preds)
    disflu_preds = disflu_preds[1:len(tokens)+1]
    
    result = []
    # print(len(tokens))
    # print(len(preds))
    assert len(tokens) == len(disflu_preds)
    for token, disflu_pred in zip(tokens, disflu_preds):
        if disflu_pred in [0,5,6]:
            result.append(token)
    # print(''.join(result))
    end_time = time.perf_counter()
    # 计算运行时间（秒）并转换为毫秒
    elapsed_time_ms = (end_time - start_time) * 1000
    result = f"输入：{text} 输出:{''.join(result)} 运行时间: {elapsed_time_ms:.2f} 毫秒"
    print(result)
    return ''.join(result)

if __name__ == '__main__':
    # 对吧？对对，2个都是现有客户呢，那就不行啊啊，我不能为为为了a客户啊，把b客户给得罪了吧。
    # text = "因为台为为为你想每个人一台嘛1400个了，就一千四百台嘛啊，每年都一一台。c++"
    text = "因为我们也做商标商标申请，商标标呃专利申请啊，所以这这个这特定的专业组开始可能需要需要特定的这个写呃那个一些数据库和系统啊，那这些可能都是软件啊。"
    # ctinfer = CTInfer(model, tokenizer)
    # result = ctinfer.infer(text)

    predict(text)
    # data = open("1726035525.5445561.test.txt","r",encoding='utf-8').readlines()
    # fw = open("ct_result.txt","w",encoding='utf-8')
    # for d in data:
    #     d = d.rstrip()
    #     d = [d[i:i+30] for i in range(0, len(d), 30)]
    #     result = ""
    #     for j in d:
    #         print(j)
    #         result += predict(j)
    #     fw.write(result+"\n")
    # fw.close()

    