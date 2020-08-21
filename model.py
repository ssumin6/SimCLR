import torch 
import torch.nn as nn
from torchvision.models import resnet18

class SimCLR(nn.Module):
    def __init__(self, out_dim):
        super(SimCLR, self).__init__()
        resnet = resnet18()
        res_out_dim = resnet.fc.in_features
        self.f = nn.Sequential(*list(resnet.children())[:-1])
        self.g = nn.Sequential(nn.Linear(res_out_dim, res_out_dim), nn.ReLU(), nn.Linear(res_out_dim, out_dim))

    def forward(self, xi, xj):
        x = torch.cat([xi, xj], dim=0)
        h = self.f(x)
        z = self.g(h.squeeze())
        return h, z

    def get_hidden(self, x):
        h = self.f(x)
        return h.squeeze()