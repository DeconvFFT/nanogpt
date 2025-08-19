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

# Simple data loader
class DataLoaderSmall:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open(f"data/input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f'Loaded: {len(tokens)} tokens')
        print(f'1 Epoch = {len(tokens)//B*T} batches')
        
        self.curr_idx = 0
        
    def get_next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_idx:self.curr_idx+B*T+1] # from current index to B*T+1 (+1 to get labels for last token)
        # for small datasets, no need to move to gpu/MPs as it will waste memory. 
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        
        # advance tensor to next position
        self.curr_idx += B*T # take chunks of size B*T
        
        # if loading next batch results in out of bounds, reset current index
        if self.curr_idx + (B*T + 1) > len(self.tokens):
            self.curr_idx = 0
            
        return x, y
        

def train_model(model, device:str):
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1337)
    loader = DataLoaderSmall(B=16, T=1024)
    torch.set_float32_matmul_precision('high')

    #torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-8) # 3e-4 is a good learning rate for GPT-2
    lr_config = get_configs('config/train_gpt2.json')
    optimizer = model.configure_optimizers(weight_decay=config['weight_decay'],learning_rate=lr_configp['max_lr'], device=device)

    for iter in range(100):
        t0 = time.time()
        x,y = loader.get_next_batch()
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # torch autocast to bf16 to save memory
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent the model from too large gradients
        lr = get_lr(iter, lr_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        d1 = (t1-t0)*1000
        tokens_per_sec = (loader.B*loader.T*1000)/d1
        
        print(f"Iteration {iter}, Loss: {loss.item()}| lr: {lr:.4e} | norm: {norm:.4f} |  dt: {d1:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}")
        
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
    
    

def load_model(device:str):
    model = GPT(GPTConfig())
    model.to(device)
    # model compilation. Pytorch doesn't just run in eager mode. It analyses the model and compiles it for faster execution. 
    # It's able to know what operations are needed, whats coming next and so it's able to optimize the model for faster execution. 
    model = torch.compile(model) # ~ 2.3x speedup
    return model

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
    model = load_model(device)
    train_model(model, device)