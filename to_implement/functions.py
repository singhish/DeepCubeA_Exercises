from torch import nn
import numpy as np

import torch  # for adam optimizer
from time import time  # for logging


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    input_size = 81  # dimension of 8-puzzle state
    hidden_layer_size = 250    # size of each hidden layer
    output_size = 1  # dimension of output (predicting a singular value, cost-to-go)

    # Define neural network (using Sequential API as its just a simple multilayer perceptron)
    class DNN(nn.Module):
        def __init__(self, D_in, H, D_out):
            super(DNN, self).__init__()

            self.layers = nn.Sequential(
                nn.Linear(D_in, H),  # input layer -> hidden layer 1
                nn.ReLU(),
                nn.Linear(H, H),     # hidden layer 1 -> hidden layer 2
                nn.ReLU(),
                nn.Linear(H, H),     # hidden layer 2 -> hidden layer 3
                nn.ReLU(),
                nn.Linear(H, D_out), # hidden layer 3 -> output layer
                nn.ReLU()            # cost-to-go should be non-negative
            )

        def forward(self, x):
            x = x.float()
            x = self.layers(x)
            return x

    return DNN(input_size, hidden_layer_size, output_size)


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
