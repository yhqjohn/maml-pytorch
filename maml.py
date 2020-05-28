import torch
from torch import nn
from torch.nn import functional as F
from torch import opti

def correct_fn(x, y, dim=1):
    with torch.no_grad():
        predict = F.softmax(x, dim=dim).argmax(dim=dim)
        _correct = torch.eq(predict, y).float().mean()
    return _correct.item()

def clip_grad_by_norm_(grad, max_norm):
    """
    in-place gradient clipping.
    :param grad: list of gradients
    :param max_norm: maximum norm allowable
    :return: average norm
    """

    total_norm = 0
    counter = 0
    for g in grad:
        param_norm = g.data.norm(2)
        total_norm += param_norm.item() ** 2
        counter += 1
    total_norm = total_norm ** (1. / 2)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grad:
            g.data.mul_(clip_coef)

    return total_norm / counter

class Logging:
    def __init__(self, is_logging):
        self.is_logging = is_logging

    def __enter__(self):
        self.is_logging = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_logging = False

    def __nonzero__(self):
        return self.is_logging

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, update_lr, meta_lr, update_step, update_step_test,
                 learner, criterion=nn.CrossEntropyLoss(), logging=False):
        """

        :param args:
        """
        super(Meta, self).__init__()
        self.update_lr = update_lr  #
        self.meta_lr = meta_lr  #
        self.update_step = update_step
        self.update_step_test = update_step_test
        self.logging = Logging(logging)
        self.meta_training = True
        self.log = {'corrects': [], 'losses': []}
        self.learner = learner
        self.F = learner.functional
        self.criterion = criterion
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)

    def forward(self, x):
        """

        :param x:   a batch list of few-shot learning n_way classifying task. Each of the task is a list [support_x, support_y, query_x, query_y],
        where support_x is the input of few shot learning support set, made of k_shots*n_way data, support_y is the ouput
        of a few shot learning support set, and query_x and query_y respectively stands for the input and out put of a query
        set.
        :return:    the loss of the query set classifying.
        """
        loss = torch.tensor(0.).requires_grad_().cuda()
        steps = self.update_step if self.meta_training else self.update_step_test
        if not self.logging.is_logging:
            # without logging the algorithm is quite simple and elegant
            for i in x:
                support_x, support_y, query_x, query_y = i
                fast_weights = list(self.learner.parameters())
                for j in range(steps):
                    logits = self.F(fast_weights, True, support_x)
                    fast_loss = self.criterion(logits, support_y)
                    grad = torch.autograd.grad(fast_loss, fast_weights, create_graph=True)
                    fast_weights = [a-self.update_lr*b for a, b in zip(fast_weights, grad)]
                loss += self.criterion(self.F(fast_weights, True, query_x), query_y)
            return loss / len(x)
        else:
            # log the losses and correctness
            batch_correctness = []
            batch_losses = []
            for i in x:
                support_x, support_y, query_x, query_y = i
                fast_weights = list(self.learner.parameters())
                corrects = []
                losses = []
                for j in range(steps):
                    logits = self.F(fast_weights, True, support_x)
                    fast_loss = self.criterion(logits, support_y)

                    # log the correctness
                    with torch.no_grad():
                        correct = correct_fn(self.F(fast_weights, True, query_x), query_y)
                    # correct = correct_fn(logits, support_y)
                    corrects.append(correct)
                    losses.append(fast_loss.item())

                    grad = torch.autograd.grad(fast_loss, fast_weights, create_graph=True)
                    fast_weights = [a-self.update_lr*b for a, b in zip(fast_weights, grad)]

                # log the correctness
                with torch.no_grad():
                    fast_loss = self.criterion(logits, support_y)
                    correct = correct_fn(self.F(fast_weights, True, query_x), query_y)
                # correct = correct_fn(logits, support_y)
                corrects.append(correct)
                losses.append(fast_loss.item())

                loss += self.criterion(self.F(fast_weights, True, query_x), query_y)

                # log the correctness
                batch_correctness.append(corrects)
                batch_losses.append(losses)

            # log the correctness
            self.log['corrects'] = batch_correctness
            self.log['losses'] = batch_losses
            return loss/len(x)

    def train(self, mode=True):
        self.meta_training = mode

    def eval(self):
        self.train(False)

    def accs(self):
        return torch.FloatTensor(self.log['corrects']).mean(dim=0).tolist()
