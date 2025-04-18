import torch
grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook