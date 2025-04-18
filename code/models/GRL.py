import torch
from torch.autograd  import  Function
import torch.nn as nn

class GRL_0(Function):
    def forward(self,input):
        return input
    def backward(self,grad_output):
        grad_input = grad_output.neg()
        return grad_input

class grl_func(torch.autograd.Function):
    def __init__(self):
        super(grl_func, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_variables
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GRL_class(nn.Module):
    def __init__(self, lambda_=0.):
        super(GRL_class, self).__init__()
        self.lambda_ = torch.tensor(lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = torch.tensor(lambda_)

    def forward(self, x):
        return grl_func.apply(x, self.lambda_)

