import torch
import torch.nn as nn

class MLP(nn.Module):
    "A simple multi-layer perceptron model for regression tasks."
    def __init__(self, cfg):
        super().__init__()
        layers = []
        input_size = cfg.model.input_size
        for hidden_size in cfg.model.hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(getattr(nn, cfg.model.activation)())
            layers.append(nn.Dropout(cfg.model.dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, cfg.model.output_size))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    


