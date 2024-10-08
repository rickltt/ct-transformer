import json
import torch

class CTTokenizer:
    def __init__(self, vocab_path):
        self.vocab = self.load_vocab(vocab_path)
        self.vocab_set = set(self.vocab)  # 使用集合加快查找
        self.vocab_dict = {token: idx for idx, token in enumerate(self.vocab)}  # 创建词典映射
        self.sentence_start_token = '<s>'
        self.sentence_end_token = '</s>'
        self.padding_token = '<pad>'
        self.unk_token = '<unk>'

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        return vocab

    def ids2tokens(self, token_ids):
        return [self.vocab[token_id] for token_id in token_ids]

    def tokens2ids(self, tokens):
        return [self.vocab_dict.get(token, self.vocab_dict[self.unk_token]) for token in tokens] 

    def tokenize(self, text):
        text = text.lower()
        tokenize_result = []

        # 遍历文本中的字符
        current_token = []
        for char in text:
            if char.isalpha() and char.isascii():
                current_token.append(char)  # 英文字符
            elif '\u4e00' <= char <= '\u9fa5':
                if current_token:
                    tokenize_result.append(''.join(current_token))
                    current_token = []
                tokenize_result.append(char)  # 中文字符
            elif char.isdigit():  # 处理数字
                current_token.append(char) 
            else:
                if current_token:
                    tokenize_result.append(''.join(current_token))
                    current_token = []
                if not char.isspace():
                    tokenize_result.append(char)  # 其他符号

        if current_token:  # 处理最后一个 token
            tokenize_result.append(''.join(current_token))

        # 处理 token 结果
        # return [token if token in self.vocab_set else self.unk_token for token in tokenize_result]
        return tokenize_result
    def __call__(self, text, max_length=None, is_split_into_words=None, return_pt = None):
        if is_split_into_words:
            tokens = text
        else:
            tokens = self.tokenize(text)
        
        if max_length is not None:
            tokens = tokens[:max_length - 2] if max_length and len(tokens) > max_length - 2 else tokens
            
        tokens = [self.sentence_start_token] + tokens + [self.sentence_end_token]
        attention_mask = [1] * len(tokens)  # 创建 attention mask
        
        if max_length is not None:
            padding_length = (max_length - len(tokens)) if max_length else 0
            tokens.extend([self.padding_token] * padding_length)
            attention_mask.extend([0] * padding_length)

        token_ids = self.tokens2ids(tokens)

        assert len(token_ids) == len(attention_mask)

        if return_pt:
            token_ids, attention_mask = torch.tensor(token_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)
            inputs = {
                "input_ids": token_ids, 
                "attention_mask": attention_mask
            }
            return inputs
        return token_ids, attention_mask

# if __name__ == '__main__':
#     tokenizer = Tokenizer('vocab.json')
#     print(len(tokenizer.vocab))
#     text = "Hello, 你好！This 2025 40 is a test."
#     token_ids, attention_mask = tokenizer(text, 32)
#     print(token_ids)
