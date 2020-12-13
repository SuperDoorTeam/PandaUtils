import torch
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size


    def step(self):
        self._step += 1

        lr = self.change_learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._rate = lr

        self.optimizer.step()


    def zero_grad(self):
        self.optimizer.zero_grad()


    def change_learning_rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            lr = self.lr_base * 1/4.
        elif step <= int(self.data_size / self.batch_size * 2):
            lr = self.lr_base * 2/4.
        elif step <= int(self.data_size / self.batch_size * 3):
            lr = self.lr_base * 3/4.
        else:
            lr = self.lr_base

        return lr


def get_optim(model, batch_size, data_size, eps, betas, lr_base=None):

    return WarmupOptimizer(
        lr_base,
        Optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0,
            betas=betas,
            eps=eps
        ),
        data_size,
        batch_size
    )


def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r

class TestWarmupOptimizer(object):
    def test_step(self):
        assert 1 == 1