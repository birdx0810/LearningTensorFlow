import json
import os
import random

from tqdm import trange
import torch
import pandas as pd
import numpy as np

import utils

# Set random seed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Get data
data = utils.get_anscombe_dataset()

# Set hyperparameters
learning_rate = 1e-3
epochs = 10
batch_size = 2

num_layers = 2
max_seq_len = 11
input_dim = 2
hidden_dim = 10

# Prepare Model
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb_rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.emb_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.emb_sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        H_o, H_t = self.emb_rnn(X)
        logits = self.emb_linear(H_o)
        H = self.emb_sigmoid(logits)
        return H

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.rec_rnn = torch.nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.rec_linear = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, H):
        H_o, H_t = self.rec_rnn(H)
        X_tilde = self.rec_linear(H_o)
        return X_tilde

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = torch.nn.MSELoss()

    def forward(self, X):
        H = self.encoder(X)
        X_tilde = self.decoder(H)
        loss = self.criterion(X_tilde, X)
        loss = 10*torch.sqrt(loss)
        return loss

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    iterations = int(len(data)/batch_size)
    for i in trange(iterations):
        model.zero_grad()
        # Get mini batch
        batch_xs, _ = utils.get_batch(data, i, batch_size)
        batch_xs = torch.Tensor(batch_xs)
        loss = model(batch_xs)
        loss.backward()
        opt.step()

    print(f"\nEpoch: {epoch}\tLoss: {loss}")
