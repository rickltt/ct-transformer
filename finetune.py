import os
import logging
import torch

from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tokenizer import CTTokenizer
from config import CTConfig
from model import CTTransformerForDisfluDetection

import numpy as np
import random
import argparse
from tqdm import tqdm, trange

# from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter


disflu_labels = ["O", "B-IM", "I-IM", "B-RM", "I-RM", "B-RP", "I-RP"]
# punct_labels = ["O",'，','。', '？']
# punct_labels = ["O",'，','。','？','、']
# punct_labels = ["O",'COMMA','PERIOD','QUESTION','ENUM']

writer = SummaryWriter('runs/finetune')

logger = logging.getLogger(__name__)

def collate_fn(batch):
    new_batch = { key: [] for key in batch[0].keys()}
    for b in batch:
        for key in new_batch:
            new_batch[key].append(torch.tensor(b[key])) 
    for b in new_batch:
        new_batch[b] = torch.stack(new_batch[b])
    return new_batch

# 定义数据集
class CTDataset(Dataset):
    def __init__(self, config, data_dir, tokenizer, max_length):

        self.data = open(data_dir,"r",encoding="utf-8").readlines()
        # self.tokens, self.disflu_labels, self.punct_labels = self.read_data(self.data)
        self.tokens, self.disflu_labels = self.read_data(self.data)
        self.disflu2id = config.disflu2id
        # self.punct2id = config.punct2id
        self.num_disflu_labels = config.num_disflu_labels
        # self.num_punct_labels = config.num_punct_labels

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = config.pad_token_id # 0 
        self.pad_label_id = config.pad_label_id # -100

    def read_data(self, data):
        tokens, disflu_labels = [], []
        # tokens, disflu_labels, punct_labels = [], [], []
        for sent in tqdm(data):
            sent = eval(sent)
            # text, disflu, punct = [], [], []
            text, disflu = [], []
            for token_label in sent:
                # token, disflu_tag, punct_tag = token_label.split("|")
                token, disflu_tag = token_label.split("|")
                text.append(token)
                disflu.append(disflu_tag)
                # punct.append(punct_tag)
            tokens.append(text)
            disflu_labels.append(disflu)
            # punct_labels.append(punct)
        # return tokens, disflu_labels, punct_labels
        return tokens, disflu_labels
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        disflu_labels = self.disflu_labels[idx]
        # punct_labels = self.punct_labels[idx]

        # assert len(tokens) == len(disflu_labels) == len(punct_labels)
        assert len(tokens) == len(disflu_labels)
        
        tokens = tokens[:self.max_length - 2] if len(tokens) > self.max_length - 2 else tokens
        disflu_labels = disflu_labels[:self.max_length - 2] if len(disflu_labels) > self.max_length - 2 else disflu_labels
        # punct_labels = punct_labels[:self.max_length - 2] if len(punct_labels) > self.max_length - 2 else punct_labels

        disflu_ids = [ self.disflu2id[i] for i in disflu_labels]
        # punct_ids = [ self.punct2id[i] for i in punct_labels]

        tokens = [self.tokenizer.sentence_start_token] + tokens + [self.tokenizer.sentence_end_token]
        disflu_ids = [self.pad_label_id] + disflu_ids + [self.pad_label_id]
        # punct_ids = [self.pad_label_id] + punct_ids + [self.pad_label_id]
        attention_mask = [1] * len(tokens)

        padding_length = self.max_length - len(tokens)

        tokens.extend([self.tokenizer.padding_token] * padding_length)
        attention_mask.extend([0] * padding_length)

        token_ids = self.tokenizer.tokens2ids(tokens)

        disflu_ids.extend([self.pad_label_id]*(self.max_length-len(disflu_ids)))
        # punct_ids.extend([self.pad_label_id]*(self.max_length-len(punct_ids)))

        assert len(token_ids) == len(attention_mask) == len(disflu_ids) == self.max_length

        inputs = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "disflu_ids": disflu_ids,
            # "punct_ids": punct_ids,
        }

        return inputs

def train(args, model, train_dataset, dev_dataset):

    args.train_batch_size = args.per_gpu_train_batch_size 

    # 采样器
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    best_f1 = 0.0
    training_loss = 0

    model.zero_grad()
    epoch_iterator = trange(0, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for epoch in epoch_iterator:

        pbar = tqdm(train_dataloader, desc="Training")

        for step, batch in enumerate(pbar):
            model.train()
            token_ids = batch['token_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            disflu_ids = batch['disflu_ids'].to(args.device)
            # punct_ids = batch['punct_ids'].to(args.device)

            # outputs = model(token_ids, attention_mask, disflu_ids, punct_ids)
            outputs = model(token_ids, attention_mask, disflu_ids)
            loss = outputs[0]
            writer.add_scalar('train_loss', loss, global_step)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                            
            loss.backward()
            training_loss += loss.item()

            epoch_iterator.set_description('Epoch: {}, Loss: {}'.format(epoch+1, round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                                     
                optimizer.step()
                scheduler.step()    
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        f1, eval_loss, _ = evaluate(args, model, dev_dataset)
                        writer.add_scalar('f1', f1, global_step)
                        writer.add_scalar('eval_loss', eval_loss, global_step)
                        if best_f1 < f1:
                            best_f1 = f1
                            if not os.path.exists(args.output_dir):
                                os.makedirs(args.output_dir)
                            save_dir = os.path.join(args.output_dir, "best_checkpoint.pt")
                
                            torch.save(model.state_dict(), save_dir)
                            logger.info("Saving best checkpoint to %s", save_dir)
                
        if args.evaluate_after_epoch:
            f1, _, _ = evaluate(args, model, dev_dataset)

            if best_f1 < f1:
                best_f1 = f1
                if not os.path.exists(args.output_dir):
                    os.makedirs(args.output_dir)
                save_dir = os.path.join(args.output_dir, "best_checkpoint.pt")
                torch.save(model.state_dict(), save_dir)
                
                logger.info("Saving best checkpoint to %s", save_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break

    return global_step, training_loss / global_step

# 定义评估函数
def evaluate(args, model, eval_dataset):

    args.eval_batch_size = args.per_gpu_eval_batch_size 
    # 采样器
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, sampler=test_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    pbar = tqdm(test_dataloader, desc="Testing")
    model.eval()
    eval_loss = 0.0
    eval_steps = 0
    disflu_preds, disflu_trues = None, None
    # punct_preds, punct_trues = None, None
    with torch.no_grad():
        for batch in pbar:
            # Move to device
            token_ids = batch['token_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            disflu_ids = batch['disflu_ids'].to(args.device)
            # punct_ids = batch['punct_ids'].to(args.device)

            # outputs = model(token_ids, attention_mask, disflu_ids, punct_ids)

            outputs = model(token_ids, attention_mask, disflu_ids)
            # disflu_logits, punct_logits = outputs[1:]
            disflu_logits = outputs[1]
            tmp_eval_loss = outputs[0]

            eval_loss += tmp_eval_loss.item()
            eval_steps += 1

            # if disflu_preds is None and punct_preds is None:
            if disflu_preds is None:
                disflu_preds = disflu_logits.detach().cpu().numpy()
                disflu_trues = disflu_ids.detach().cpu().numpy()

                # punct_preds = punct_logits.detach().cpu().numpy()
                # punct_trues = punct_ids.detach().cpu().numpy()
            else:
                disflu_preds = np.append(disflu_preds, disflu_logits.detach().cpu().numpy(), axis=0)
                disflu_trues = np.append(disflu_trues, disflu_ids.detach().cpu().numpy(), axis=0)

                # punct_preds = np.append(punct_preds, punct_logits.detach().cpu().numpy(), axis=0)
                # punct_trues = np.append(punct_trues, punct_ids.detach().cpu().numpy(), axis=0)

    disflu_preds = np.argmax(disflu_preds, axis=2)          
    # punct_preds = np.argmax(punct_preds, axis=2)

    disflu_true_predictions = [
        [disflu_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(disflu_preds, disflu_trues)
    ]
    disflu_true_labels = [
        [disflu_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(disflu_preds, disflu_trues)
    ]

    # punct_true_predictions = []
    # for prediction, label in zip(punct_preds, punct_trues):
    #     for (p, l) in zip(prediction, label):
    #         if l != -100:
    #             punct_true_predictions.append(punct_labels[p])

    # punct_true_labels = []
    # for prediction, label in zip(punct_preds, punct_trues):
    #     for (p, l) in zip(prediction, label):
    #         if l != -100:
    #             punct_true_labels.append(punct_labels[l])

    # punct_list = [ l for l in punct_labels if l!= "O"]

    # punct_per_precision, punct_per_recall, punct_per_f1, _  = precision_recall_fscore_support(punct_true_labels, punct_true_predictions, average=None, labels = punct_list)
    # punct_overall = precision_recall_fscore_support(punct_true_labels, punct_true_predictions, average='micro', labels = punct_list )

    # punct_overall_result = pd.DataFrame(
    #     np.array([punct_per_precision, punct_per_recall, punct_per_f1]),
    #     columns=punct_list,
    #     index=['Precision', 'Recall', 'F1'])
    
    # punct_overall_result['OVERALL'] = punct_overall[:3]
    # punct_overall_result = punct_overall_result.round(2)
    # print(punct_overall_result)

    # punct_precision, punct_recall, punct_f1 = punct_overall[0], punct_overall[1], punct_overall[2]
    # logger.info(f"punct_p:{punct_precision}, punct_r:{punct_recall}, punct_f1:{punct_f1}")


    disflu_precision=precision_score(disflu_true_labels, disflu_true_predictions)
    disflu_recall=recall_score(disflu_true_labels, disflu_true_predictions)
    disflu_f1=f1_score(disflu_true_labels, disflu_true_predictions)
    disflu_overall_result = classification_report(disflu_true_labels, disflu_true_predictions)
    print(disflu_overall_result)
    
    logger.info(f"disflu_p:{disflu_precision}, disflu_r:{disflu_recall}, disflu_f1:{disflu_f1}")

    # overall_result = str(punct_overall_result) + "\n\n" + str(disflu_overall_result)
    # f1 = (punct_f1 + disflu_f1) / 2

    return disflu_f1, eval_loss / eval_steps, disflu_overall_result

# 设置随机种子
def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/finetune_data", type=str)
    parser.add_argument("--model_path", default="./output/pretrain_output/model.pt", type=str)

    parser.add_argument("--output_dir", default='./output/finetune_output/', type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--evaluate_after_epoch", action="store_true",
                        help="Whether to run evaluation after each epoch.")
    parser.add_argument("--per_gpu_train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--dropout_prob", default=0.1, type=float,
                        help="dropout_prob.")

    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=str, default='0.5',
                        help="Log every X updates steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    
    args = parser.parse_args()
    return args

# 主程序
if __name__ == "__main__":

    args = parse_args()
    # 设置随机种子

    if os.path.exists(args.output_dir):
        os.system(f"rm -rf {args.output_dir}")
        os.system(f"rm -rf ./run/finetune")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)

    set_seed(args)

    config = CTConfig('config.json')
    # 加载Tokenizer和模型
    tokenizer = CTTokenizer('vocab.json')    

    disflu2id = {j:i for i,j in enumerate(disflu_labels)}
    # punct2id = {j:i for i,j in enumerate(punct_labels)}
    num_disflu_labels = len(disflu_labels)
    # num_punct_labels = len(punct_labels)    

    config.disflu2id = disflu2id
    # config.punct2id = punct2id
    config.num_disflu_labels = num_disflu_labels
    # config.num_punct_labels = num_punct_labels

    model = CTTransformerForDisfluDetection(config)
    model.from_pretrained(args.model_path)
    model.to(args.device)

    train_data_path = os.path.join(args.data_dir,'train.txt')
    dev_data_path = os.path.join(args.data_dir,'dev.txt')
    test_data_path = os.path.join(args.data_dir,'test.txt')

    train_dataset = CTDataset(config, train_data_path, tokenizer, args.max_seq_length)
    dev_dataset = CTDataset(config, dev_data_path, tokenizer, args.max_seq_length)
    test_dataset = CTDataset(config, test_data_path, tokenizer, args.max_seq_length)

    if args.do_train:  
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        save_dir = os.path.join(args.output_dir, "last_checkpoint.pt")
        torch.save(model.state_dict(), save_dir)
        logger.info("Saving last checkpoint to %s", save_dir)
                            
    
        # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint.pt')

        state_dict = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(args.device)

        f1, _, overall_result = evaluate(args, model, test_dataset)
        output_eval_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_eval_file, "a") as writer:
            writer.write('***** Predict in test dataset ***** \n')
            writer.write("{} \n".format(overall_result))