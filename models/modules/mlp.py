from torch import nn


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hid_features,
                 out_features,
                 dropout=0.):
        super(MLP, self).__init__()

        out_features = out_features or in_features
        hid_features = hid_features or in_features

        self.fc1 = nn.Linear(in_features, hid_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hid_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

if __name__ == '__main__':
    import torch

    x = torch.randn(1,197,768)
    model = MLP(768, 3096, 1024)

    # (1,197,768) -> (1,197,3096) -> (1,197,1024)
    print(model(x).size()) # torch.Size([1,197,1024])
