import os
import logging
import torch

from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tokenizer import CTTokenizer
from config import CTConfig
from model import CTTransformerForPreTraining
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import argparse
from glob import glob
from tqdm import tqdm, trange

writer = SummaryWriter('runs/pretrain')

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
    def __init__(self, config, tokenizer, max_length):

        self.tokens, self.disflu_labels, self.punct_labels = self.read_data(config.data_dir)

        self.disflu2id = config.disflu2id
        self.punct2id = config.punct2id
        self.num_disflu_labels = config.num_disflu_labels
        self.num_punct_labels = config.num_punct_labels


        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_label_id = config.pad_label_id

    def read_data(self, data_dir):
        data_files = glob(os.path.join(data_dir,"*.txt"))
        texts, disflu_labels, punct_labels = [], [], []
        for file in data_files:
            lines = open(file,"r",encoding="utf-8").readlines()
            logger.info(f"Loading {file} data!")
            for line in tqdm(lines):
                sent = eval(line)
                text, disflu, punct = [], [], []
                for token_label in sent:
                    if token_label.count("|") > 2:
                        continue
                    token, disflu_tag, punct_tag = token_label.split("|")
                    text.append(token)
                    disflu.append(disflu_tag)
                    punct.append(punct_tag)
                texts.append(text)
                disflu_labels.append(disflu)
                punct_labels.append(punct)

        assert len(texts) == len(disflu_labels) == len(punct_labels)
        return texts, disflu_labels, punct_labels
    
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        disflu_labels = self.disflu_labels[idx]
        punct_labels = self.punct_labels[idx]

        tokens = tokens[:self.max_length - 2] if len(tokens) > self.max_length - 2 else tokens
        disflu_labels = disflu_labels[:self.max_length - 2] if len(disflu_labels) > self.max_length - 2 else disflu_labels
        punct_labels = punct_labels[:self.max_length - 2] if len(punct_labels) > self.max_length - 2 else punct_labels

        disflu_ids = [ self.disflu2id[i] for i in disflu_labels]
        punct_ids = [ self.punct2id[i] for i in punct_labels]

        tokens = [self.tokenizer.sentence_start_token] + tokens + [self.tokenizer.sentence_end_token]
        disflu_ids = [self.pad_label_id] + disflu_ids + [self.pad_label_id]
        punct_ids = [self.pad_label_id] + punct_ids + [self.pad_label_id]
        attention_mask = [1] * len(tokens)

        padding_length = self.max_length - len(tokens)

        tokens.extend([self.tokenizer.padding_token] * padding_length)
        attention_mask.extend([0] * padding_length)

        token_ids = self.tokenizer.tokens2ids(tokens)

        disflu_ids.extend([self.pad_label_id]*(self.max_length-len(disflu_ids)))
        punct_ids.extend([self.pad_label_id]*(self.max_length-len(punct_ids)))

        assert len(token_ids) == len(attention_mask) == len(disflu_ids) == len(punct_ids) == self.max_length

        inputs = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "disflu_ids": disflu_ids,
            "punct_ids": punct_ids,
        }

        return inputs

def train(args, model, train_dataset):

    train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    
    # 采样器
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, sampler=train_sampler, batch_size=train_batch_size)

    # 初始化优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=args.lr)

    # t_total = len(train_dataloader) //  args.epochs
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=0, num_training_steps=t_total
    # )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    # logger.info("  Total optimization steps = %d", t_total)


    training_loss = 0
    global_step = 0
    model.zero_grad()
    epoch_iterator = trange(0, int(args.epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    for epoch in epoch_iterator:

        pbar = tqdm(train_dataloader, desc="Training", disable=args.local_rank not in [-1, 0])

        for batch in pbar:
            model.train()
            token_ids = batch['token_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            disflu_ids = batch['disflu_ids'].to(args.device)
            punct_ids = batch['punct_ids'].to(args.device)

            outputs = model(token_ids, attention_mask, disflu_ids, punct_ids)
            loss = outputs[0]
            writer.add_scalar('Loss', loss, epoch)

            epoch_iterator.set_description('Epoch: {}, Loss: {}'.format(epoch+1, round(loss.item(), 6)))

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            training_loss += loss.item()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()                                         
            model.zero_grad()
            global_step += 1

        if args.local_rank in [-1, 0] and (epoch+1)%5==0:
            model_to_save = model.module if hasattr(model, "module") else model 
            model_to_save = model_to_save.ct_tranformer
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            save_path = os.path.join(args.output_dir, f"epoch:{epoch+1}_loss:{training_loss / global_step}.pt")

            torch.save(model_to_save.state_dict(), save_path)

    if args.local_rank in [-1, 0]:
        model_to_save = model.module if hasattr(model, "module") else model 
        model_to_save = model_to_save.ct_tranformer
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        save_path = os.path.join(args.output_dir, "model.pt")

        torch.save(model_to_save.state_dict(), save_path)

    logger.info(f"Training loss: {training_loss / global_step}")


# 设置随机种子
def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    ## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--no_cuda", action="store_true", help="Whether to cuda.")
    parser.add_argument("--seed", default=2024, type=int)
    
    parser.add_argument("--data_dir", default="./pretrain_data", type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--output_dir", default="./outputs/", type=str)
    
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--per_gpu_batch_size", default=600, type=int)
    parser.add_argument("--max_length", default=256, type=int)

    args = parser.parse_args()
    return args

# 主程序
if __name__ == "__main__":

    args = parse_args()
    # 设置随机种子

    if os.path.exists(args.output_dir):
        os.system(f"rm -rf {args.output_dir}")
        os.system("rm -rf ./run/pretrain")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    set_seed(args)

    config = CTConfig('config.json')
    # 加载Tokenizer和模型
    tokenizer = CTTokenizer('vocab.json')

    disflu_labels = ["O", "B-IM", "I-IM", "B-RM", "I-RM", "B-RP", "I-RP"]
    punct_labels = ["O",'，','。','？','、']

    disflu2id = {j:i for i,j in enumerate(disflu_labels)}
    punct2id = {j:i for i,j in enumerate(punct_labels)}
    num_disflu_labels = len(disflu_labels)
    num_punct_labels = len(punct_labels)

    config.disflu2id = disflu2id
    config.punct2id = punct2id
    config.num_disflu_labels = num_disflu_labels
    config.num_punct_labels = num_punct_labels


    model = CTTransformerForPreTraining(config)
    if args.model_path is not None:
        model.from_pretrained(args.model_path)
    model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total number of parameters: {num_params}')
    config.train_data = os.path.join(args.data_dir,'data.txt')
    # config.data = open(args.data_dir,"r",encoding='utf-8').readlines()
    config.data_dir = args.data_dir
    pretrain_dataset = CTDataset(config, tokenizer, args.max_length)
    train(args, model, pretrain_dataset)
    
    writer.close()

