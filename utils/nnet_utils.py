import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import re
from torch import Tensor

from time import time  # for logging

def states_nnet_to_pytorch_input(states_nnet: np.ndarray, device) -> Tensor:
    states_nnet_tensor = torch.tensor(states_nnet, device=device)

    return states_nnet_tensor


def train_nnet(nnet: nn.Module, states_nnet: np.ndarray, outputs: np.ndarray, batch_size: int, num_itrs: int,
               train_itr: int):
    print_skip = 100

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(nnet.parameters())
    batch_start_idx = 0
    batch_start_time = time()

    for itr in range(train_itr, train_itr + num_itrs):
        # get batch of training examples
        start_idx = batch_start_idx
        end_idx = batch_start_idx + batch_size
        input_batch = torch.tensor(states_nnet[start_idx:end_idx])
        target_batch = torch.tensor(outputs[start_idx:end_idx])

        # complete pass over batch
        pred_batch = nnet(input_batch)
        loss = loss_fn(target_batch, pred_batch)
        if itr % print_skip == 0:  # print loss every 100 training iterations
            print(f"Itr: {itr}, "
                  f"loss: {round(loss.item(), 5)}, "
                  f"targ_ctg: {round(target_batch.float().mean().item(), 2)}, "
                  f"nnet_ctg: {round(pred_batch.float().mean().item(), 2)}, "
                  f"Time: {round(time() - batch_start_time, 2)}")
            batch_start_time = time()
        loss.backward()

        # update optimizer
        optimizer.step()
        optimizer.zero_grad()

        # increment to next batch
        batch_start_idx += batch_size


# loading nnet
def load_nnet(model_file: str, nnet: nn.Module, device: torch.device = None) -> nn.Module:
    # get state dict
    if device is None:
        state_dict = torch.load(model_file)
    else:
        state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet
