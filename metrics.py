import torch
import numpy as np


def psnr(target, prediction):
    x = (target-prediction)**2
    x = x.view(x.shape[0], -1)
    p = torch.mean((-10/np.log(10))*torch.log(torch.mean(x, 1)))
    return p
