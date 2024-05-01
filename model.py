import torch.nn as nn
import torch

class L1(torch.nn.Module):
    def __init__(self, module, beta=0.01):
        super().__init__()
        self.module = module
        self.beta = beta


    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return -1*self.beta * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)

class model(torch.nn.Module):
    def __init__(self, module, weight_decay):
        def __init__(self):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Conv2d(3,32),
                nn.Linear(28*28, 512),
                L1(),
                nn.ReLU(),
                nn.Linear(512, 512),
                L1(),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits