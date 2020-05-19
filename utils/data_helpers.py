import numpy as np
import torch
from torch.utils.data import Dataset


def convert_to_tensor(number):
    binary_num = list(map(int, format(number, 'b')))
    return torch.tensor(binary_num).to(torch.float)


def num_from_tensor(tens):
    bin_list = tens.view(-1).tolist()
    return sum([2 ** i * int(bin_list[- i - 1]) for i in range(len(bin_list))])


####################################################
#############  MULTIPLES OF N DATASET  #############
####################################################

def gen(kind, size, start=0):
    multipliers = np.arange(size)
    if kind == 'even':
        multipliers = 2 * multipliers
    elif kind == 'odd':
        multipliers = 2 * multipliers + 1
    else:
        raise Exception("kind must be 'even' or 'odd'")
    multipliers += start
    return list(multipliers)


def get_data(size, start):
    half_size = size // 2
    evens = gen('even', half_size, start)
    odds = gen('odd', half_size, start)

    return evens + odds, [1] * half_size + [0] * half_size


class EvenOddDataset(Dataset):
    def __init__(self, size, start=0):
        inputs, targets = get_data(size, start)
        inputs = [convert_to_tensor(i).view(-1, 1, 1) for i in inputs]
        targets = torch.tensor(targets)
        self.data = list(zip(inputs, targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample(self):
        idx = np.random.randint(0, len(self.data))
        return self.data[idx][0], self.data[idx][1].view(1)

####################################################
####################################################
####################################################