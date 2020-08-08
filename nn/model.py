import torch
import torch.nn as nn

class LSTMModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units):
        super(LSTMModel, self).__init__()
        
        self.lstm_units = lstm_units
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.LN1 = torch.nn.LayerNorm(embedding_dim)
        self.LSTM = torch.nn.LSTM(embedding_dim, lstm_units, batch_first=True, dropout=0.3)
        self.LN2 = torch.nn.LayerNorm(lstm_units)
        self.LSTM2 = torch.nn.LSTM(lstm_units, lstm_units, batch_first=True, dropout=0.5)
        self.LN3 = torch.nn.LayerNorm(lstm_units)
        self.Linear = torch.nn.Linear(lstm_units, vocab_size)
    
    def forward(self, x, state_h, state_c):
        x = self.embedding(x)
        x = self.LN1(x)
        x, (state_h, state_c) = self.LSTM(x, (state_h, state_c))
        x = self.LN2(x)
        x, (state_h, state_c) = self.LSTM2(x, (state_h, state_c))
        x = self.LN3(x)
        x = self.Linear(x)
        return x, state_h, state_c
    
    def reset_state(self, batch_size, device):
        return torch.zeros(1, batch_size, self.lstm_units, device=device), torch.zeros(1, batch_size, self.lstm_units, device=device)