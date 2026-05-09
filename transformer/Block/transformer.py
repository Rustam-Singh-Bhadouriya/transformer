import torch
import torch.nn as nn
from transformer.components import (MultiHeadAttention,
                        FeedForwardNetwork,
                        EmbeddingLayer,
                        PositionalEncoding,
                        causal_mask
                         )
from torch.utils.data import DataLoader
import torch.optim as optim



class TransformerBlock(nn.Module):
    def __init__(self, n_heads:int, d_model : int = 128, d_ff : int = 512, dropout : float=0.1):
        super().__init__()


        self.attention = MultiHeadAttention(
            n_heads=n_heads,
            d_model=d_model
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(
            d_ff=d_ff,
            d_model=d_model,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):

        # Multi Head Attention

        attention_out  = self.attention(x, mask)

        x = self.norm1(
            x + self.dropout(attention_out)
        )

        ffn_out = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ffn_out)
        )

        return x

        
class Transformer(nn.Module):
    def __init__(self, vocab_size : int, d_model : int = 128, d_ff : int = 512, n_heads : int = 4, n_layers : int = 4, max_len : int = 512, dropout : float = 0.1 ):
        super().__init__()

        self.Embedder = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)

        self.postionalEncoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(n_heads=n_heads, d_model=d_model, d_ff=d_ff, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        mask = causal_mask(T)

        x = self.Embedder(x)

        x = self.postionalEncoder(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)

        logits = self.projection(x)

        return logits


if __name__ == "__main__":
    data_raw = torch.tensor([
    [1,2,3,4],
    [5,6,7,8],
    [2,4,6,8],
    [0,1,2,3]
    ])

    loader = DataLoader(
        dataset=data_raw,
        batch_size=2,
        shuffle=True
    )

    model = Transformer(vocab_size=9, max_len=32)

    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # training loop
    for epoch in range(10):

        for batch in loader:

            # input
            x = batch[:, :-1]

            # target
            y = batch[:, 1:]

            # forward pass
            logits = model(x)

            # reshape for cross entropy
            loss = loss_fn(
                logits.reshape(-1, 9),
                y.reshape(-1)
            )

            # clear old gradients
            optimizer.zero_grad()

            # compute gradients
            loss.backward()

            # update weights
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item()}")
    
    tokens = torch.tensor([[1,2,3]])

    for _ in range(5):

        logits = model(tokens)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1
        )

    print(tokens)