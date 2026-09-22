import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class NASController(nn.Module):
    def __init__(self, num_layers=16, num_choices=5, hidden_size=64):
        super(NASController, self).__init__()
        self.num_layers = num_layers
        self.num_choices = num_choices
        
        # LSTM cell handles the sequential generation of the architecture
        self.lstm = nn.LSTMCell(input_size=num_choices, hidden_size=hidden_size)
        
        # Maps the LSTM's hidden state to unnormalized probabilities (logits) for the channel choices
        self.decoder = nn.Linear(hidden_size, num_choices)
        
        # A learnable starting token to kick off the sequence generation
        self.init_input = nn.Parameter(torch.zeros(1, num_choices))
        
    def forward(self):
        log_probs = []
        actions = []
        
        # Setup initial hidden and cell states
        h_t = torch.zeros(1, self.lstm.hidden_size, device=self.init_input.device)
        c_t = torch.zeros(1, self.lstm.hidden_size, device=self.init_input.device)
        x = self.init_input
        
        for _ in range(self.num_layers):
            h_t, c_t = self.lstm(x, (h_t, c_t))
            logits = self.decoder(h_t)
            
            # Create a probability distribution from the logits and sample an index
            dist = Categorical(logits=logits)
            action = dist.sample()
            
            # Store the log probability for the REINFORCE algorithm update
            log_probs.append(dist.log_prob(action))
            actions.append(action.item())
            
            # The next step's input is the one-hot encoded choice of the current step
            x = F.one_hot(action, num_classes=self.num_choices).float()
            
        # Return the sequence of sampled indices and the sum of their log probabilities
        return actions, torch.stack(log_probs).sum()