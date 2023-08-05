"""Simple bigram model training"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One Head Self Attention"""

    def __init__(self, head_size: int, n_embed: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.values = nn.Linear(n_embed, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        # compute queries, keys, values
        queries = self.query(x)  # (B, T, C)
        keys = self.key(x)  # (B, T, C)
        values = self.values(x)  # (B, T, C)

        # compute dot products, scale, mask, and softmax
        wei = queries @ keys.transpose(-2, -1) * C ** (-0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        wei = self.dropout(wei)

        # apply attention
        out = wei @ values  # (B, T, C)

        return out


class MultiHeadAttention(nn.Module):
    """Multi Head Self Attention"""

    def __init__(
        self, n_heads: int, n_embed: int, head_size: int, dropout: float = 0.1
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embed=n_embed) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(head_size * n_heads, head_size * n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is (B, T, C)
        heads = [h(x) for h in self.heads]  # [(B, T, C)] * n_heads
        out = torch.cat(heads, dim=-1)  # (B, T, C * n_heads)
        out = self.proj(out)  # (B, T, C * n_heads)
        out = self.dropout(out)  # (B, T, C * n_heads)

        return out


class FeedForward(nn.Module):
    """Feed Forward Layer"""

    def __init__(self, n_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_hidden, n_hidden * 4),
            nn.ReLU(),
            nn.Linear(n_hidden * 4, n_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(
            n_heads=n_heads,
            head_size=head_size,
            dropout=dropout,
            n_embed=n_embed,
        )
        self.feed_forward = FeedForward(n_hidden=n_embed, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=n_embed)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=n_embed)

    def forward(self, x):
        # x is (B, T, C)
        x = x + self.sa_heads(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, n_layer: int, n_heads: int, n_embed: int, dropout: float = 0.1):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        self.attention_blocks = nn.Sequential(
            *[
                AttentionBlock(n_embed=n_embed, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.layer_norm_final = nn.LayerNorm(normalized_shape=n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embds = self.token_embedding_table(idx)  # (B,T,n_embed)
        position_embds = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T,n_embed)

        # add the two embeddings
        embds = token_embds + position_embds  # (B,T,n_embed)
        embds = self.attention_blocks(embds)  # (B,T,n_embed)
        embds = self.layer_norm_final(embds)  # (B,T,n_embed)
        logits = self.lm_head(embds)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


if __name__ == "__main__":
    _n_layer = 6
    _n_heads = 6
    _dropout = 0.2
    _n_embed = 384

    _max_iters = 5000
    _eval_iters = 200
    _eval_interval = 500
    _learning_rate = 3e-4

    model = BigramLanguageModel(
        n_heads=_n_heads, n_layer=_n_layer, n_embed=_n_embed, dropout=_dropout
    )
    model = model.to(device)

    # print number of parameters in the model
    print(f"Running on {device}")
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=_learning_rate)

    tqdm_loop = tqdm(range(_max_iters))
    for iter in tqdm_loop:

        # every once in a while evaluate the loss on train and val sets
        if iter % _eval_interval == 0:
            losses = estimate_loss(eval_iters=_eval_iters)
            tqdm_loop.set_postfix(
                text=f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_idxs = model.generate(context, max_new_tokens=500)[0].tolist()

    # print the generated sequence
    decoded_text = decode(generated_idxs)
    print(decoded_text)
