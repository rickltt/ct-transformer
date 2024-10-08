import random
import json
import jieba
import re
import time
import os
import pandas as pd
import string
import threading
from tqdm import tqdm
from glob import glob
from tokenizer import CTTokenizer
# chinese_punctuation = "，。！？；：（）【】《》“”‘’、"
# punctuation_list = string.punctuation + chinese_punctuation
punctuation_list = '！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
interregnum_list = ['吗', '吧', '呀', '呃', '呐', '呗', '呢', '呵', '哇', '哉', '哎', '哩', '唔', '唸', '啦', '啵', '嘛', '嘞', '欤', '啊', '哦', '恩', '嗯']

tokenizer = CTTokenizer("vocab.json")

punct_labels = ['，','。', '？']

def split_list(lst, n):
    """
    将列表 lst 切分成 n 份，如果不能整除则最后一份可能会稍短。
    :param lst: 待切分的列表
    :param n: 切分的份数
    :return: 切分后的列表集合
    """
    # 计算每一份的长度
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def create_example(sentence):
    sentence = sentence.lstrip(punctuation_list)
    k_rm = random.randint(1,5)
    k_im = random.randint(1,3)
    k = k_rm + k_im
    if not sentence:
        return None
        # raise TypeError(
        #     """A 'NoneType' object received while a 'str' object is required."""
        # )
    else:
        words = jieba.lcut(sentence)
        words = list(filter(None, words))
        # words = [word for word in words if word not in [' ','']]
        # assert '' not in words and ' ' not in words
        # words = tokenizer.tokenize(sentence)
    
    words_without_rm_im = [ word for word in words if word not in punctuation_list and word not in interregnum_list ]
    if len(words_without_rm_im) < k:
        return None

    indices = []
    all_indexes = list(range(len(words)))
    random.shuffle(all_indexes)
    count = 0
    idx = 0
    while count < k:
        if words[all_indexes[idx]] not in punctuation_list and words[all_indexes[idx]] not in interregnum_list:
            indices.append(all_indexes[idx])
            count+=1
        idx+=1
    assert len(indices) == k

    # selected_words = [words[i] for i in indices]
    # print(f"selected_words:{selected_words}")

    rm_indices = indices[:k_rm]
    im_indices = indices[k_rm:]
    tokens = []
    disflu_tags = []
    for idx,word in enumerate(words):
        if word == ' ':
            continue
        if idx in rm_indices:
            if len(word) == 1:
                options = [1, 2, 3]
                weights = [0.4, 0.35, 0.25] 
            else:
                options = [1, 2]
                weights = [0.5, 0.5] 
            choices = random.choices(options, weights=weights, k=1)[0]
            if word.isalpha() and word.isascii():
                new_word = [ word ] * choices
                new_word = ' '.join(new_word)
            else:  
                new_word = word * choices
            
            # start = time.time()
            new_word = tokenizer.tokenize(new_word)
            # end = time.time()
            # print("tokenize时间:%.2f秒"%(end-start))
            tokens.extend(new_word)
            disflu_tags.append("B-RM")                
            disflu_tags.extend(["I-RM"]*(len(new_word)-1))

            word = tokenizer.tokenize(word)   
            tokens.extend(word)
            disflu_tags.append("B-RP") 
            disflu_tags.extend(["I-RP"]*(len(word)-1)) 
            # print(f"words:{words}")
            # print(f"word:{word}")
            # print(tokens)
            # print(labels)
            assert len(tokens) == len(disflu_tags)

        elif idx in im_indices:
            word = tokenizer.tokenize(word)   
            tokens.extend(word)
            disflu_tags.extend(["O"]*len(word))
            im_word = random.choice(interregnum_list)
            im_word = tokenizer.tokenize(im_word)   
            tokens.extend(im_word)
            disflu_tags.append("B-IM")
            disflu_tags.extend(["I-IM"]*(len(im_word)-1))
            assert len(tokens) == len(disflu_tags)
        else:
            word = tokenizer.tokenize(word) 
            tokens.extend(word)
            disflu_tags.extend(["O"]*len(word))
            assert len(tokens) == len(disflu_tags)
    
    # punct_tags = []
    # for idx in range(1, len(tokens)):
    #     next_token = tokens[idx]
    #     if next_token in punctuation_list:
    #         punct_tags.append(next_token)
    #     else:
    #         punct_tags.append("O")
    # punct_tags.append("O")

    # punct_tags = [ punct if punct in punct_labels else "O" for punct in punct_tags  ]
    # assert len(tokens) == len(disflu_tags) == len(punct_tags)
    assert len(tokens) == len(disflu_tags)
    # token_label = [ token + "|" + disflu_tag + "|" + punct_tag for token, disflu_tag, punct_tag in zip(tokens,disflu_tags,punct_tags) if token not in punctuation_list]
    token_label = [ token + "|" + disflu_tag for token, disflu_tag in zip(tokens,disflu_tags)]
    # print(token_label)
    return token_label

def create_iwslt2012(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in ["train","dev","test"]:
        data = open(os.path.join(data_dir, f"{split}.txt"),"r",encoding="utf-8").readlines()
        token_label_data = []
        for sentence in tqdm(data):
            sentence = sentence.replace(" ","").replace("\n","")
            sentence = remove_non_chinese_english_chars(sentence)

            token_label = create_example(sentence)
            if token_label == None:
                continue
            token_label_data.append(token_label)

        with open(os.path.join(output_dir, f"{split}.txt"),"w",encoding="utf-8") as f:
            for i in token_label_data:
                f.write(str(i)+"\n")   
                
def create_csc(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in ["train","dev","test"]:
    # for split in ["test"]:
        data = json.load(open(os.path.join(data_dir, f"{split}.json"),"r",encoding="utf-8"))
        # data = [ d["correct_text"] for d in data]
        # new_data = []
        # for i in range(0,len(data),2):
        #     if i + 1 < len(data):
        #         new_data.append(data[i] + data[i+1])
        #     else:
        #         new_data.append(data[i])
        # data = [ data[i] + data[i+1] for i in range(0,len(data),2)]
        token_label_data = []
        for item in tqdm(data):
            # sentence = "地处武林路，相当繁华的地带啊，内部环境不错，很干净，地上的头发会当即清扫。发型师的服务态度也很好，当然价格是比较高的，"
            # sentence = item["correct_text"]
            # sentence = process_text(sentence)
            if item["wrong_ids"] == []:
                token_label = [ token+"|O|O"  for token in sentence if token not in punctuation_list]
            else:
            #     # start = time.time()
                sentence = item["correct_text"]
                sentence = remove_non_chinese_english_chars(sentence)
                token_label = create_example(sentence)
            #     # end = time.time()
            #     # print("create_repetitions时间:%.2f秒"%(end-start))
                if token_label == None:
                    continue
                token_label_data.append(token_label)
            # sentence = remove_non_chinese_english_chars(sentence)
            # token_label = create_example(sentence)
            # if token_label == None:
            #     continue
            # token_label_data.append(token_label)

        random.shuffle(token_label_data)
        with open(os.path.join(output_dir, f"{split}.txt"),"w",encoding="utf-8") as f:
            for i in token_label_data:
                f.write(str(i)+"\n")    

def remove_non_chinese_english_chars(text):
    # 定义要保留的字符范围
    chinese_characters = r'\u4e00-\u9fff' # 基本汉字
    english_characters = r'a-zA-Z'
    punctuation = re.escape('''!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~，。！？、；：“”‘’《》〈〉【】〔〕（）‥·°〃※》々◦〝〞〃〄〆''')

    # 正则表达式模式
    pattern = f'[^{chinese_characters}{english_characters}{punctuation}]+'
    
    # 使用正则表达式替换非保留字符
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def create_baike(data, output_dir, output_name):

    token_label_data = []
    for sentence in tqdm(data):
        token_label = create_example(sentence)
        if token_label == None:
            continue
        token_label_data.append(token_label)
    with open(os.path.join(output_dir, f"{output_name}.txt"),"w",encoding="utf-8") as f:
        for i in token_label_data:
            f.write(str(i)+"\n")   

class BaikeThread(threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, data, output_dir):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data = data
        self.output_dir = output_dir

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print("Starting:",self.name)
        create_baike(self.data, self.output_dir, self.name)
        print("Exiting:",self.name)

def baike(data_dir, max_length_per_sent, output_dir):
    # data_dir = "./dataset/baike2018qa/baike_qa_train.json"
    data = []
    lines = open(data_dir,"r",encoding="utf-8").readlines()
    
    for line in lines:

        line = eval(line)
        answer = line["answer"]
        answer = answer.replace("\r","").replace("\n","")
        sentence = remove_non_chinese_english_chars(answer)

        split_lines= [sentence[i:i+max_length_per_sent] for i in range(0, len(sentence), max_length_per_sent)]
        for part in split_lines:
            data.append(part)
    # data = data[:300]
    print(f"data size: {len(data)}")

    thread_num = 30
    baike_dataset = split_list(data, thread_num)

    for i in range(thread_num):
        thread = BaikeThread(i+1, f"baike_{i+1}", baike_dataset[i], output_dir)
        thread.start() 


class WikiThread(threading.Thread):   #继承父类threading.Thread
    def __init__(self, threadID, name, data_dir, max_length_per_sent, output_dir):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.data_dir = data_dir
        self.max_length_per_sent = max_length_per_sent
        self.output_dir = output_dir

    def run(self):                   #把要执行的代码写到run函数里面 线程在创建后会直接运行run函数 
        print("Starting:",self.name)
        create_wiki(self.data_dir, self.max_length_per_sent, self.output_dir, self.name)
        print("Exiting:",self.name)

def wiki(data_dir, max_length_per_sent, output_dir):
    # wiki_dir = "./dataset/wiki_zh"
    wiki_files = os.listdir(data_dir)
    thread_num = len(wiki_files)
    # output_dir = "./pretrain_data"

    for i in range(thread_num):
        thread = WikiThread(i+1, wiki_files[i], os.path.join(data_dir,wiki_files[i]), max_length_per_sent, output_dir)
        thread.start()

def create_wiki(data_dir, max_length_per_sent, output_dir, output_name):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    token_label_data = []

    files_dir = glob(data_dir+"/*")
    for file in tqdm(files_dir):
        lines = open(file,"r",encoding="utf-8").readlines()
        # lines = lines[:10]
        for line in lines:
            line = eval(line)
            text = line["text"]
            docs = text.split("\n\n")
            # print(docs)
            sentences = []
            for doc in docs:
                # print(doc)
                if len(doc) < 10:
                    continue
                doc = remove_non_chinese_english_chars(doc)
                doc = doc.replace("\n","").replace(" ","")
                doc = doc.lstrip()
                doc = doc.rstrip()

                split_lines= [doc[i:i+max_length_per_sent] for i in range(0, len(doc), max_length_per_sent)]
                for part in split_lines:
                    sentences.append(part)
            for sentence in sentences:
                # print(sentence)
                token_label = create_example(sentence)
                if token_label == None:
                    continue
                token_label_data.append(token_label)

    with open(os.path.join(output_dir, f"{output_name}.txt"),"w",encoding="utf-8") as f:
        for i in token_label_data:
            f.write(str(i)+"\n")   

if __name__ == "__main__":
    # sentence = "，！。Hello， 你好！This 2025、 40 is (a) test？"
    # create_example(sentence)

    # output_dir = "./pretrain_data_punct"
    # wiki_dir = "./dataset/wiki_zh"
    # max_length_per_sent = 256
    # baike_dir = "./dataset/baike2018qa/baike_qa_train.json"
    # wiki(wiki_dir, max_length_per_sent, output_dir)
    # baike(baike_dir, max_length_per_sent, output_dir)
        
    create_csc("./dataset/csc", "./dataset/finetune_csc")
    # create_iwslt2012("./dataset/iwslt2012_zh", "./dataset/finetune_punct_data")


