import torch
import torch.nn as nn

class Tactile_CNN(nn.Module):
    def __init__(self, num_features=24, mat_classes=6, tex_classes=6):
        super(Tactile_CNN, self).__init__()

        self.dual_cls = True
        
        self.hidden_size = 128  # Size of LSTM hidden state
        self.num_layers = 2     # Number of LSTM layers
        self.bidirectional = True
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # Output size is doubled if bidirectional
        lstm_out_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size

        # Fully connected layers for each classification task
        self.material_classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, mat_classes)
        )

        self.texture_classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Linear(64, tex_classes)
        )

    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last time step's output (from both directions if bidirectional)
        if self.bidirectional:
            last_output = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_output = h_n[-1]
        
        material_logits = self.material_classifier(last_output)
        texture_logits = self.texture_classifier(last_output)
        
        return material_logits, texture_logits