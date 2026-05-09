import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformer.Block import Transformer

text = """
hello Coders <END>
transformers are amazing <END>
pytorch is fun <END>
attention is powerful <END>
language models predict tokens <END>
deep learning is cool <END>
"""

words = text.split()

vocab = sorted(set(words))

stoi = {
    word:i
    for i,word in enumerate(vocab)
}

itos = {
    i:word
    for word,i in stoi.items()
}
vocab_size = len(words)


data = torch.tensor([
    stoi[word]
    for word in words
])

seq_len = 16

samples = []

for i in range(len(data) - seq_len):

    chunk = data[i:i+seq_len+1]

    samples.append(chunk)

samples = torch.stack(samples)

loader = DataLoader(
    samples,
    batch_size=4,
    shuffle=True
)

model = Transformer(vocab_size=vocab_size, max_len=64)

loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(10):
    for batch in loader:

        x = batch[:, :-1]

        y = batch[:, 1:]

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")

start = "transformers"

tokens = torch.tensor([[
    stoi[word]
    for word in start.split()
]])

# generation loop
for _ in range(20):

    # forward pass
    logits = model(tokens)

    # take last token prediction
    next_token_logits = logits[:, -1, :]

    # predict next token
    next_token = torch.argmax(
        next_token_logits,
        dim=-1,
        keepdim=True
    )

    if next_token.item() == stoi["<END>"]:
        break

    # append prediction
    tokens = torch.cat(
        [tokens, next_token],
        dim=1
    )

    

# decode tokens back to text
decoded = " ".join([
    itos[i]
    for i in tokens[0].tolist()
])

print(decoded)

# Star if you like :)