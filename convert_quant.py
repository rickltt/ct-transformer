import onnx
import os
import torch
import time
import random
import onnxruntime
import numpy as np
from pprint import pprint
from config import CTConfig
from tokenizer import CTTokenizer
from model import CTTransformerForDisfluDetection

from onnxruntime.quantization import quantize_dynamic, QuantType

config = CTConfig("./config.json")
tokenizer = CTTokenizer("./vocab.json")

disflu_labels = ["O", "B-IM", "I-IM", "B-RM", "I-RM", "B-RP", "I-RP"]
num_disflu_labels = len(disflu_labels)

config.num_disflu_labels = num_disflu_labels

model = CTTransformerForDisfluDetection(config)

model_path = "./output/finetune_output_disflu/best_checkpoint.pt"

state_dict = torch.load(model_path, weights_only=True, map_location="cuda:0")
model.load_state_dict(state_dict)
model.eval()


def generate_inputs(batch_size, max_len):

    input_ids = []
    for _ in range(batch_size):
        seq_length = random.randint(5,10)
        token_ids = np.random.randint(1,config.vocab_size,(seq_length,))
        token_ids = [1] + token_ids.tolist() + [2]
        token_ids = np.array(token_ids)
        token_ids = np.append(token_ids,np.zeros(max_len-len(token_ids)),axis=0)
        input_ids.append(torch.tensor(token_ids).long())   

    input_ids = torch.stack(input_ids)
    attention_mask = (input_ids != 0).long()
    print(input_ids)
    print(attention_mask)
    return input_ids, attention_mask

def export_onnx(output_dir):
    # 导出模型为 ONNX 格式
    
    batch_size = 1  #批处理大小
    sequence_length = 256

    dynamic_axes= {"input_ids":{0:"batch_size",1:"sequence_length"},	
                    "attention_mask":{0:"batch_size",1:"sequence_length"},
                    "logits":{0:"batch_size",1:"sequence_length"},
    }
    token_ids, attention_mask = generate_inputs(batch_size, sequence_length)
    dummy_input = {
        "input_ids": token_ids,
        "attention_mask":attention_mask
    }
    torch.onnx.export(model, dummy_input, os.path.join(output_dir,'model.onnx'), 
                      export_params=True,
                      training=torch.onnx.TrainingMode.EVAL, 
                    #   opset_version=15, 
                      do_constant_folding=True,
                      input_names=['input_ids','attention_mask'], 
                      output_names=['logits'],
                      dynamic_axes=dynamic_axes,
                      verbose=True)

def quantize(model_fp32, model_quant):
    quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QInt8)
    
def printinfo(onnx_session):
    print("----------------- 输入部分 -----------------")
    input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
    for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表

        input_info = {
            "name" : input_tensor.name,
            "type" : input_tensor.type,
            "shape": input_tensor.shape,
        }
        pprint(input_info)

    print("----------------- 输出部分 -----------------")
    output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
    for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表

        output_info = {
            "name" : output_tensor.name,
            "type" : output_tensor.type,
            "shape": output_tensor.shape,
        }
        pprint(output_info)

if __name__ == "__main__":
    output_dir = "./output/finetune_output_disflu"
    # export_onnx(output_dir)

    # model_fp32 = os.path.join(output_dir,"model.onnx")
    # model_quant = os.path.join(output_dir,"model_quant.onnx")
    # quantize(model_fp32, model_quant)
    
    # model = onnx.load(os.path.join(output_dir,"model_quant.onnx"))
    # onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))
    ort_session = onnxruntime.InferenceSession(os.path.join(output_dir,"model_quant.onnx"))
    printinfo(ort_session)
    
    # batch_size = 1  #批处理大小
    # sequence_length = 256

    # token_ids, attention_mask = generate_inputs(batch_size, sequence_length)
    start_time = time.perf_counter()
    # text = "因为台为为为你想每个人一台嘛1400个了，就一千四百台嘛啊，每年都一一台。c++语言运行速度好快。"
    text = "当当然可能可能刚好就那个呃，他那边对也也也也是我的客户，那我不能代表您去告他，对不对？那对对对对，这就有一冲了，这才是这个概念。"
    tokens = tokenizer.tokenize(text)
    token_ids, attention_mask = tokenizer(text)
    # print(token_ids)
    # print(attention_mask)
    outputs = ort_session.run(None, {
        "input_ids": np.array(token_ids).reshape(-1,len(token_ids)),
        "attention_mask": np.array(attention_mask).reshape(-1,len(token_ids)),
    })
    disflu_preds = outputs[0].argmax(-1).reshape(-1).tolist()
    disflu_preds = disflu_preds[1:len(tokens)+1]
    result = []
    print(disflu_preds)
    assert len(tokens) == len(disflu_preds)
    for token, disflu_pred in zip(tokens, disflu_preds):
        if disflu_pred in [0,5,6]:
            result.append(token)

    end_time = time.perf_counter()
    # 计算运行时间（秒）并转换为毫秒
    elapsed_time_ms = (end_time - start_time) * 1000
    result = f"输入：{text} 输出:{''.join(result)} 运行时间: {elapsed_time_ms:.2f} 毫秒"
    print(result)
    
