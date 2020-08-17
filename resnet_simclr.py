import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self,out_dim, device):
        super(ResNetSimCLR, self).__init__()
        resnet = models.resnet18(pretrained=False).to(device)
        num_ftrs = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs).to(device)
        self.l2 = nn.Linear(num_ftrs, out_dim).to(device)


    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x
