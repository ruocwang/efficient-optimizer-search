from torch import nn


class LinearNet(nn.Module): ## a regression model for quadratic loss
    def __init__(self, input_dim=10, hidden_dim=10, n_classes=1, bias=False):
        super(LinearNet, self).__init__()
        self.fc = nn.Linear(hidden_dim, n_classes, bias=bias)
        self.init()

    def init(self): ## match that of learning2learn
        self.fc.weight.data.fill_(0)

    def forward(self, x):
        x = self.fc(x)
        return x
