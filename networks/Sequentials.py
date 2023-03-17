import torch
import torch.nn as nn
import math


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        self.lstm.flatten_parameters() 
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[-1]
        
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        self.gru.flatten_parameters() 
        
        output, hn = self.gru(x, h0)
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[-1]
        
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        
        out = self.sigmoid(out)
        return out       


class Bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Bi_LSTM, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        
        self.lstm.flatten_parameters()
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        hn = hn[-1]
        
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out


class LSTM_drop(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_drop, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.dropout = nn.Dropout(0.25)
        
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        self.lstm.flatten_parameters() 
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[-1]
        
        out = self.relu(hn)
        out = self.dropout(out)
        
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class LSTM_fc(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM_fc, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc_2 =  nn.Linear(128, 64)
        self.fc = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        self.lstm.flatten_parameters() 
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[-1]
        
        out = self.relu(hn)
        
        out = self.fc_1(out)
        out = self.relu(out)
        
        out = self.fc_2(out)
        out = self.relu(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class Conv_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(Conv_LSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        
        self.conv1_1 = nn.Conv1d(in_channels=input_size, out_channels = 64, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels = 32, kernel_size=3, stride=1)
        
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        
        x = x.transpose(1,2)   # shape: (batch_size, 12, 22) -> (batch_size, 22, 12)
        
        x = self.conv1_1(x) # (batch_size, 22, 12) -> (batch_size, 64, 12)
        
        x = self.conv1_2(x) # (batch_size, 64, 12) -> (batch_size, 32, 12)
        
        x = x.transpose(1, 2)  # (batch_size, 32, 12) -> (batch_size, 12, 32)
        
        self.lstm.flatten_parameters() 
        
        output, (hn, cn) = self.lstm(x)
        
        hn = hn[-1]
        
        out = self.relu(hn)
        
        out = self.fc_1(out)
        out = self.relu(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out



class BiLSTM_fc(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM_fc, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc_2 =  nn.Linear(128, 64)
        self.fc = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)
        
        
        self.lstm.flatten_parameters() 
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[-1]
        
        out = self.relu(hn)
        
        out = self.fc_1(out)
        out = self.relu(out)
        
        out = self.fc_2(out)
        out = self.relu(out)
        
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out

# +
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransformerEncoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # input x = (batch_size, seq_length, input_size) = (batch_size, 12, 22)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        attn_mask = torch.zeros(x.size(0), x.size(0), device=self.device).bool()
        attn_mask = attn_mask.masked_fill(torch.triu(attn_mask) == 1, False)
        
        output = self.transformer_encoder(x, src_key_padding_mask=None, mask=attn_mask)
        output = output.permute(1, 0, 2)
        output = output.mean(dim=1)
        
        out = F.relu(self.fc_1(output))
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
# -


