import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
from gpt2 import GPTConfig, GPT
import json

# autodetect device
def detect_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS")
        return 'mps'
    else:
        print("Using CPU")
        return 'cpu'
def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt
# Simple data loader
class DataLoaderSmall:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in ['train', 'val'], "split must be either train or val"
        
        # get shards
        data_dir = "data/fineweb/edu_fineweb10B"
        shards = os.listdir(data_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_dir, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split: {split}"
        if master_process:
            print(f"Loading {split} data from {len(shards)} shards")
        
        self.reset()
        
        
        
    def get_next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_idx:self.curr_idx+B*T+1] # from current index to B*T+1 (+1 to get labels for last token)
        # for small datasets, no need to move to gpu/MPs as it will waste memory. 
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        
        # advance tensor to next position
        self.curr_idx += B*T*self.num_processes # take chunks of size B*T*num_processes
        
        # if loading next batch results in out of bounds, reset current index
        if self.curr_idx + (B*T*self.num_processes + 1) > len(self.tokens):
            # advance to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            # advance to next chunk
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.curr_idx = self.B * self.T * self.process_rank
        return x, y
    
    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.curr_idx = self.B * self.T * self.process_rank
def check_ddp_env():
    from torch.distributed import init_process_group, destroy_process_group
    
    # setting up ddp
    # torchun command sets the env variables RANK, LOCALRANK AND WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")
    
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process
    
def train_model(model,raw_model, ddp, ddp_rank, ddp_local_rank, ddp_world_size, ddp_device, master_process, device):

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1337)
    torch.set_float32_matmul_precision('high')


    # Gradient accumulation
    total_batch_size = 524288 # 2**19 ~ .5M tokens
    B = 16
    T = 1024
    assert total_batch_size % (B*T*ddp_world_size) == 0 # Make sure total batch size is divisible by (B*T*ddp_world_size)
    grad_accumulation_steps = total_batch_size // (B*T*ddp_world_size)
    if master_process:
        print(f" Total batch size: {total_batch_size}, Batch size: {B}, Sequence length: {T}, Grad accumulation steps: {grad_accumulation_steps}")

    loader = DataLoaderSmall(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

    #torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8) # 3e-4 is a good learning rate for GPT-2
    lr_config = get_configs('config/train_gpt2.json')
    optimizer = raw_model.configure_optimizers(weight_decay=lr_config['weight_decay'],learning_rate=lr_config['max_lr'],betas=(lr_config['beta1'], lr_config['beta2']), device=device)

    for iter in range(100):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0
        # accumulate gradients
        for micro_step in range(grad_accumulation_steps):
            x,y = loader.get_next_batch()
            x,y = x.to(device), y.to(device)
            
            # torch autocast to bf16 to save memory
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x,y)
            loss = loss/grad_accumulation_steps # scale the loss to account for the gradient accumulation. If we don't do this, loss = sum(lossi), we loose on the normalization factor 
            loss_accum+=loss.detach()
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accumulation_steps-1) # only sync gradients on the last micro_step
            loss.backward() # loss.backward() accumulates gradients so it does a +=
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent the model from too large gradients
        lr = get_lr(iter, lr_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)
        tokens_processed = loader.B*loader.T*grad_accumulation_steps*ddp_world_size
        tokens_per_sec = (tokens_processed)/dt
        if master_process:
            print(f"Iteration {iter}, Loss: {loss_accum.item():.6f}| lr: {lr:.4e} | norm: {norm:.4f} |  dt: {dt*1000:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}")
    if ddp:
        destroy_process_group()
        
def get_configs(filename:str):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def get_lr(step:int,config:dict):
    # Learning rate scheduler
    warmup_steps = config['warmup_steps']
    max_steps = config['max_steps']
    max_lr = config['max_lr']
    min_lr = config['min_lr']
    
    # 1.) Linear warmup till model reaches warmup_steps
    if step < warmup_steps:
        return max_lr * (step+1)/warmup_steps
    # 2.) If we've reached max_steps, return min_lr
    if step>max_steps:
        return min_lr
    # 3.) In between, using cosine decay down to min_lr
    # decay learning rate
    decay_ratio = (step-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    # coeff is a value between 0 and 1 that determines the amount of decay
    coeff = 0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr + coeff*(max_lr-min_lr)
    
    

def load_model(device:str, ddp, ddp_local_rank):
    model = GPT(GPTConfig())
    model.to(device)
    # model compilation. Pytorch doesn't just run in eager mode. It analyses the model and compiles it for faster execution. 
    # It's able to know what operations are needed, whats coming next and so it's able to optimize the model for faster execution. 
    model = torch.compile(model) # ~ 2.3x speedup
    if ddp:
        # once backward pass is done, the gradients are all reduced and averaged across the world. 
        # sync gradients across the world. 
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model
    return model,raw_model

def generate_from_pretrained(device:str):
    n_return_sentences = 5
    max_length = 30
    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    ## replicating what we did in the notebook
    ## prefix tokens:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2") # get encoding for GPT-2

    ## encode the input text
    tokens = enc.encode("Hello, I'm a language model,") # encode the input text
    tokens = torch.tensor(tokens, dtype=torch.long, device=device) #(8,)
    tokens = tokens.unsqueeze(0).repeat(n_return_sentences, 1) # shape: (5,8)
    x = tokens.to(device)

    ## generate new tokens.x.shape = (B,T) -> B = 5, T = 8
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1337)
    

    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)
            logits = logits[:, -1, :] # (B,T,vocab_size) -> (B,vocab_size)
            
            # softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # top 50 sampling
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            
            # sample from the distribution
            # avoid sampling very rare tokens
            # helps keep model on track
            idx = torch.multinomial(topk_probs, num_samples=1)
            
            # gather corresponding indices to the sampled tokens
            xcol = torch.gather(topk_indices, -1, idx)
            x = torch.cat((x, xcol), dim=1)


    # print geenrated text
    for i in range(n_return_sentences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

#generate_from_pretrained(device = detect_device())

if __name__ == "__main__":
    device = detect_device()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, ddp_device, master_process = check_ddp_env()

    model,raw_model = load_model(device, ddp, ddp_local_rank)
    train_model(model,raw_model,ddp, ddp_rank, ddp_local_rank, ddp_world_size, ddp_device, master_process, device)