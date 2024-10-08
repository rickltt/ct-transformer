import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class CTEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.register_buffer('position_embeddings', self._get_sinusoid_encoding_table(config.max_position_embeddings, config.d_model))
        self.LayerNorm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            # this part calculate the position In brackets
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        # [:, 0::2] are all even subscripts, is dim_2i
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        word_embeddings = self.word_embeddings(input_ids)
        embeddings = word_embeddings + self.position_embeddings[:, :word_embeddings.size(1)]
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CTSelfAttention(nn.Module):
    def __init__(self, config, max_future_length):
        super().__init__()
        self.max_future_length = max_future_length
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.d_model / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.d_model, self.all_head_size)
        self.key = nn.Linear(config.d_model, self.all_head_size)
        self.value = nn.Linear(config.d_model, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def my_triu(self, x: torch.tensor, diagonal: int):
        l = x.size(-1)
        arange = torch.arange(l)
        mask = arange.expand(l,l)
        mask = mask-diagonal

        arange = arange.unsqueeze(-1)
        mask = torch.le(mask, arange)
        
        return mask
    
    def generate_ct_mask(self, batch_size: int, seq_len: int):
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        # mask = torch.triu(mask, diagonal=self.max_future_length)
        mask = self.my_triu(mask, diagonal=self.max_future_length)
        mask = torch.stack([mask] * batch_size)
        mask = mask.reshape(batch_size,1,seq_len,seq_len)
        return mask

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)


    def expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        # inverted_mask = 1.0 - expanded_mask

        # return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        return expanded_mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]
        ct_mask = self.generate_ct_mask(batch_size, seq_length)
        attention_mask = self.expand_mask(attention_mask, attention_mask.dtype)
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        attention = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        ct_mask = ct_mask.to(attention_mask.device)
        attention = attention.masked_fill(ct_mask == 0, float("-1e20"))
        attention = attention.masked_fill(attention_mask == 0, float("-1e20"))
        
        attention = attention / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # context_layer即attention矩阵与value矩阵的乘积，原始的大小为：(batch_size, num_attention_heads, sequence_length, attention_head_size) ；
        context_layer = torch.matmul(attention_probs, value_layer)

        # context_layer进行转置和view操作以后，形状就恢复了(batch_size, sequence_length, hidden_size)。
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs)
        
        return outputs


class CTLayer(nn.Module):
    def __init__(self, config, max_future_length):
        super().__init__()
        self.attention = CTSelfAttention(config, max_future_length)

        self.linear1 = nn.Linear(config.d_model, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear2 = nn.Linear(config.intermediate_size, config.d_model)

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:

        # FFN(x) = max(0, x*W_1+ b_1)*W_2 + b_2 
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        return self.dropout2(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
        )
        attention_scores = self_attention_outputs[0]

        x = hidden_states
        
        # 残差
        x = self.norm1(x + self.dropout1(attention_scores))
        
        # FeedForward
        x = self.norm2(x + self._ff_block(x))
  
        return x

class CTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_1 =  [CTLayer(config,0) for _ in range(config.num_hidden_layers-1)]
        self.layer_2 = [CTLayer(config,config.L_N)]
        
        self.layers = nn.ModuleList(self.layer_1 + self.layer_2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        all_hidden_states = ()
        for layer_module in self.layers:

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
            )

            hidden_states = layer_outputs

            all_hidden_states = all_hidden_states + (hidden_states,)
        last_hidden_state = hidden_states
        output = (last_hidden_state, all_hidden_states)
        return output


class CTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = CTEmbeddings(config)
        self.encoder = CTEncoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        embedding_output = self.embeddings(input_ids)
        
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
        )
        
        last_hidden_state = encoder_outputs[0]

        return last_hidden_state


class CTTransformerForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_disflu_labels = config.num_disflu_labels
        self.num_punct_labels = config.num_punct_labels
        self.ct_tranformer = CTTransformer(config)
        self.disflu_tagging_layer = nn.Linear(config.d_model, self.num_disflu_labels)
        self.punct_tagging_layer = nn.Linear(config.d_model, self.num_punct_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        self.ct_tranformer.load_state_dict(state_dict) 

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        disflu_labels: Optional[torch.Tensor] = None,
        punct_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        last_hidden_state = self.ct_tranformer(input_ids, attention_mask)
        last_hidden_state = self.dropout(last_hidden_state)

        disflu_logits = self.disflu_tagging_layer(last_hidden_state)
        punct_logits = self.punct_tagging_layer(last_hidden_state)


        loss = None
        if disflu_labels is not None and punct_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            disflu_loss = loss_fct(disflu_logits.view(-1, self.num_disflu_labels), disflu_labels.view(-1))
            punct_loss = loss_fct(punct_logits.view(-1, self.num_punct_labels), punct_labels.view(-1))
            loss = disflu_loss + punct_loss

        outputs = (loss, disflu_logits, punct_logits)

        return outputs 

class CTTransformerForDisfluDetection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_disflu_labels = config.num_disflu_labels
        # self.num_punct_labels = config.num_punct_labels
        self.ct_tranformer = CTTransformer(config)
        self.disflu_tagging_layer = nn.Linear(config.d_model, self.num_disflu_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        self.ct_tranformer.load_state_dict(state_dict) 

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        disflu_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        last_hidden_state = self.ct_tranformer(input_ids, attention_mask)
        last_hidden_state = self.dropout(last_hidden_state)

        disflu_logits = self.disflu_tagging_layer(last_hidden_state)

        loss = None
        if disflu_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(disflu_logits.view(-1, self.num_disflu_labels), disflu_labels.view(-1))
            
        outputs = (loss, disflu_logits)

        return outputs 

class CTTransformerForPunct(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_punct_labels = config.num_punct_labels
        self.ct_tranformer = CTTransformer(config)
        self.punct_tagging_layer = nn.Linear(config.d_model, self.num_punct_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def from_pretrained(self, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        self.ct_tranformer.load_state_dict(state_dict) 

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        punct_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        
        last_hidden_state = self.ct_tranformer(input_ids, attention_mask)
        last_hidden_state = self.dropout(last_hidden_state)

        punct_logits = self.punct_tagging_layer(last_hidden_state)

        loss = None
        if punct_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(punct_logits.view(-1, self.num_punct_labels), punct_labels.view(-1))

        outputs = (loss, punct_logits)

        return outputs 
    
# def generate_inputs(batch_size, max_len):

#     input_ids = []
#     for _ in range(batch_size):
#         seq_length = random.randint(5,10)
#         token_ids = np.random.randint(1,config.vocab_size,(seq_length,))
#         token_ids = np.append(token_ids,np.zeros(max_len-len(token_ids)),axis=0)
#         input_ids.append(torch.tensor(token_ids).long())   

#     input_ids = torch.stack(input_ids)
#     attention_mask = (input_ids != 0).long()

#     return input_ids, attention_mask

# if __name__ == '__main__':
#     config = CTConfig('config.json')

#     # batch_size = 4 
#     max_len = 32  
#     from tokenizer import Tokenizer
#     tokenizer = Tokenizer('vocab.json')
#     text = "Hello, 你好！This is a test."
#     input_ids, attention_mask = tokenizer.encode(text, max_len)
#     input_ids, attention_mask = torch.tensor(input_ids).unsqueeze(0), torch.tensor(attention_mask).unsqueeze(0)
#     # input_ids, attention_mask = generate_inputs(batch_size, max_len)
#     model = CTTransformer(config)
#     outputs = model(input_ids, attention_mask)
