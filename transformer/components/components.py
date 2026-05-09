import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model : int = 128, layer_norm=True):
        super().__init__()
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, max_norm=1.0)

        self.layer_norm = layer_norm
        if self.layer_norm: # Layer Norminization if `True`
            self.layerNorm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.embedder(x)
        if self.layer_norm:
            return self.layerNorm(x)
        
        return x



# class PositionalEncoding(nn.Module):
#     def __init__(self):
#         self.encoder = nn.

def causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 128, max_len: int = 5000):
        super().__init__()
        
        # 1. Create a matrix of [max_len, d_model] to hold encoding values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 2. Compute the div_term (10000^(2i/d_model)) in log space for stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 3. Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 4. Add batch dimension and register as buffer (won't be updated by optimizer)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Add encoding to embeddings up to the current sequence length
        return x + self.pe[:, :x.size(1), :]


class SelfAttention(nn.Module):
    def __init__(self, head_dim : int = 128, ):
        super().__init__()
        self.head_dim = head_dim

        self.Wq = nn.Linear(head_dim, head_dim)
        self.Wk = nn.Linear(head_dim, head_dim)
        self.Wv = nn.Linear(head_dim, head_dim)
    
    def forward(self, x=None, mask=None):
        # x should be like [B, T, C]
        # B -> Batch size, T -> length, C -> embedding_dim or d_model or head_dim

        if x is None:
            raise ValueError("Empty or Invalid parameter `x`")

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        #  Single Head Attention Score

        score = Q @ K.transpose(-2, -1)

        # scaling - for Stability

        score = score / math.sqrt(self.head_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        
        # probability
        attention = torch.softmax(score, dim=-1)

        # weighted sum on probs
         
        out = attention @ V

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads : int, d_model : int = 128):
        super().__init__()

        assert d_model % n_heads == 0  # if n_heads is not divisible with d_model

        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

        self.heads = nn.ModuleList([
            SelfAttention(head_dim=self.head_dim)
            for _ in range(n_heads)
        ])

        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        B, T, C = x.shape

        x = x.view(B, T, self.n_heads, self.head_dim)

        output = []

        for i, head in enumerate(self.heads):
            head_i = x[:, :, i, :]

            head_output = head(head_i, mask)

            output.append(head_output)
        

        out = torch.cat(output, dim=-1)

        out = self.projection(out)

        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_ff:int=None,  d_model : int = 128, dropout : float = 0.1):
        super().__init__()

        if d_ff is None:
            d_ff = d_model * 4
        
        self.layer1 = nn.Linear(d_model, d_ff)

        self.activation = nn.GELU()

        self.layer2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.layer2(x)

        return x
        
        

if __name__ == "__main__":
    data_raw = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [2,4,6,8],
    [0,1,2,3]
])

    data = DataLoader(
        dataset=data_raw,
        batch_size=2,
        shuffle=True
    )

    embedder = EmbeddingLayer(11)

    positionEncoder = PositionalEncoding(
        d_model=128,
        max_len=11
    )

    for batch in data:

        print("Batch:", batch)
        print("Batch Shape:", batch.shape)

        x = embedder(batch)

        print("Embedding Shape:", x.shape)

        positional_x = positionEncoder(x)

        print("Positional Shape:", positional_x.shape)

        mask = causal_mask(4)

        singleHeadAttention = SelfAttention()
        attention = singleHeadAttention(x, mask=mask)

        print("Single Head Attention: ", attention)
        print("Single Head Attention Shape: ", attention.shape)

        Multiattention = MultiHeadAttention(n_heads=2)
        attention = Multiattention(x, mask=mask)

        print("Attention: ", attention)
        print("Attention Shape: ", attention.shape)
        print("-" * 30)

