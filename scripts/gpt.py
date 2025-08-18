""" Version 1: GPT Model: Bigram
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_heads = 6
n_layers = 6
dropout_rate = 0.2

torch.manual_seed(1337)


with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str)->tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y  = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """A single head of self attention
    """
    def __init__(self, head_size:int)->None:
        """ Initialize network  params
        
        Args:
            head_size (int): Size of single attention head
        """
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # buffer is a tensor that is not a parameter of the model and is not a part of the state_dict
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Pass input through the network

        Args:
            x (torch.Tensor): Inputs to the network

        Returns:
            torch.Tensor: Output of the network
        """
        B,T,C = x.shape
        k = self.key(x) # (B,T,C = head_size)
        q = self.query(x) # (B,T,C = head_size)
        v = self.value(x) # (B, T, C = head_size)
        
        # calculate attention scores
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T]==0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        out = weights @ v #(B, T, T) @ (B, T, C) -> (B, T, C = head_size)
        return out
    
class MultiAttentionHead(nn.Module):
    """ Multiple self attention heads in parallel"""
    
    def __init__(self, num_heads:int, head_size:int)->None:
        """ Initialise Params for Multi headed self attention

        Args:
            num_heads (int): Number of heads we want
            head_size (int): Dimension of each head
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embed, n_embed) # Linear transformation of the outcome
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        """ Forwards network inputs

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor with multi headed self attention and a Linear projection applied to those attentions
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concat over channel dimension
        out = self.dropout(self.projection(out)) # Projection back to our original pathway
        return out 
    
class FeedForward(nn.Module): # adding a linear layer to allow tokens some time to think about what they've learned
    """ Simple Linear Layer followed by a non linearity """
    
    def __init__(self, n_embed:int)->None:
        """ Initialise parameters and network for a simple Feed forward network

        Args:
            n_embed (int): Embedding size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4* n_embed),
            nn.ReLU(),
            nn.Linear(4* n_embed, n_embed), # projection layer
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        """ Pass input tensors through network

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Tensor with a linear transformation and RELU activation applied and projected back to original pathway
        """
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    
    def __init__(self, n_embed:int, n_heads:int):
        """_summary_

        Args:
            n_embed (int): Embedding size
            n_heads (int): Number of heads we want
        """
        super().__init__()
        head_size = n_embed//n_heads 
        self.self_attention = MultiAttentionHead(num_heads=n_heads, head_size=head_size)
        self.feed_forward = FeedForward(n_embed=n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed) # Layer norm is similar to batch norm but it normalises rows instead of columns. And we don't need to keep track of running mean and variances
        self.layer_norm2 = nn.LayerNorm(n_embed) # we apply layer norm on inputs of self attention and feed forward
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """_summary_
        Args: x: Input Tensor
        Returns:
            torch.Tensor: Tensor with self attention passed through a Feed Forward network
        """
        x = x + self.self_attention(self.layer_norm1(x)) # x =  x +  (any other computation): adds a residual connection for (any other computation). We deviate from the original path and add the results back in once done
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
    
class GPTLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)# Embedding identity of tokens
        self.positional_embedding_table = nn.Embedding(block_size, n_embed) # embedding of position of tokens. Each token from 0 to block_size-1 has a unique embedding
        self.blocks = nn.Sequential(*[Block(n_embed =n_embed, n_heads=n_heads) for _ in range(n_layers)])
        self.layer_norm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) 
        
        # weight initialisation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx:torch.Tensor, targets:torch.Tensor=None)->tuple[torch.Tensor, torch.Tensor|None]:
        #logits = self.token_embedding_table(idx) # (B,T,C) (Batch, Time, Channel)
        # adding token embeddings and then we try to get logits from them using a linear layer
        
        B,T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B,T,C = n_embed) # token embedding layer
        pos_embeddings = self.positional_embedding_table(torch.arange(T, device = device)) # (T,C = n_embed). ALl integers from 0 to T-1 have a unique embedding
        x = token_embeddings + pos_embeddings # (B,T,C = n_embed) 
        # x = self.sa_heads(x) # adding one head of self attention to embeddings (token + positional)
        # x = self.ffwd(x) # (B, T, C) # all tokens think about the data they gathered
        ## instead of using sa and ffd, we now use blocks
        x = self.blocks(x)
        x = self.layer_norm_final(x) # Apply layer norm before passing to language modeling head
        logits = self.lm_head(x) # (B,T,C = vocab_size) # Language modeling head:
        
        ## Pytorch expects inputs to be (B, C, T)
       
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx:torch.Tensor, max_new_tokens:int)->torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] 
            # predictions on indices
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    
    # every eval interaval
    if iter % eval_interval ==0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f'Step: {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
    
    # Sample a batch of data
    Xb, yb = get_batch('train')
    
    
    # evaluate loss:
    logits, loss = model(Xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # break # checkpoint break

context = torch.zeros((1,1), dtype=torch.long, device=device)

print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
save_path = 'models/gpt.pth'
save_directory = os.path.dirname(save_path)

# Create the directory if it does not exist
# The 'exist_ok=True' argument prevents an error if the directory already exists
os.makedirs(save_directory, exist_ok=True)
torch.save(model.state_dict(), save_path)

# load model and run inference:
m1 = GPTLanguageModel()
m1.load_state_dict(torch.load(save_path, weights_only=True))
m1.eval()
print(f'Saved Model Generated: {decode(m1.generate(context, max_new_tokens=500)[0].tolist())}')