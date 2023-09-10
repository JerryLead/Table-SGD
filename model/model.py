import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=16):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x

def model_to_vector(model):
    return torch.cat([param.view(-1) for param in model.parameters()], dim=0)