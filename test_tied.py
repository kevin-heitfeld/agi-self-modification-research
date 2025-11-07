import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 5)
        self.output = nn.Linear(5, 10, bias=False)
        self.output.weight = self.embed.weight

m = M()
print('Named parameters:')
for n, p in m.named_parameters():
    print(f'  {n}: ptr={p.data_ptr()}')
