import torch


def cpu():
    return torch.device('cpu')


def gpu(i=0):
    return torch.device(f'cuda:{i}')


def num_gpus():
    return torch.cuda.device_count()


def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
