import random
import numpy as np

import torch


def set_seed(random_seed):
    """Random number fixed"""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return self.val


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def is_empty(t):
    return t.nelement() == 0


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


# sampling helpers
def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs
