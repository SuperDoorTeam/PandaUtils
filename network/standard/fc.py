import torch
import torch.nn as nn

from torch.nn.utils.weight_norm import weight_norm

class MLP(nn.Module):
    def __init__(self, input_dim, dims, activation='ReLU', dropout=0., use_wnorm=False):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.dimensions = dims
        self.activation = activation
        self.dropout = dropout
        # Modules
        linears = []
        for din, dout in zip(dims[:-1], dims[1:]):
            if use_wnorm:
                linears.append(weight_norm(nn.Linear(din, dout), dim=None))
            else:
                linears.append(nn.Linear(din, dout))
            linears.append(nn.__dict__[activation]())
            if dropout > 0:
                linears.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*linears)

    def forward(self, x):
        return self.mlp(x)


class TestMLP(object):
    def test_forward(self):
        x = torch.randn([20, 10])
        dims = [10, 20, 8]
        mlp = MLP(10, dims, 'ReLU', 0.2, use_wnorm=True)
        ret = mlp.forward(x)
        assert (ret.size(1) == 8)

    def test_structure(self):
        dims = [10, 20, 8]
        mlp = MLP(10, dims, 'ReLU')

        assert mlp.mlp[0].in_features == dims[0] and mlp.mlp[0].out_features == dims[1] \
               and mlp.mlp[2].in_features == dims[1] and mlp.mlp[2].out_features == dims[2]
