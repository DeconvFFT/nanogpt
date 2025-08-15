import torch
import torch.nn as nn
import torch.nn.functional as F


# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Using device: {device}')
eval_iters = 200
n_embed = 32 


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
            X,y = get_batch(split)
            logits, loss = model(X,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx:torch.Tensor, targets:torch.Tensor=None)->tuple[torch.Tensor, torch.Tensor|None]:
        #logits = self.token_embedding_table(idx) # (B,T,C) (Batch, Time, Channel)
        # adding token embeddings and then we try to get logits from them using a linear layer
        token_embeddings = self.token_embedding_table(idx) # (B,T,C = n_embed)
        logits = self.lm_head(token_embeddings) # (B,T,C = vocab_size)
        
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
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# Optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
for iter in range(max_iters):
    
    # every eval interaval
    if iter % eval_interval ==0:
        losses = estimate_loss()
        print(f'Step: {iter}, train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')
    
    # Sample a batch of data
    Xb, yb = get_batch('train')
    
    
    # evaluate loss:
    logits, loss = model(Xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    
    
    
    
    
    
        