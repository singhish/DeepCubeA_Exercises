from torch import nn
import numpy as np

import torch
from time import time


def get_nnet_model() -> nn.Module:
    """ Get the neural network model

    @return: neural network model
    """
    input_size = 9  # dimension of 8-puzzle state
    hidden_layer_size = 250
    output_size = 1  # dimension of output (predicting a singular value, cost-to-go)

    # Define neural network (here just a simple multilayer perceptron)
    class DNN(nn.Module):

        def __init__(self, D_in, H, D_out):
            super(DNN, self).__init__()

            self.fc1 = nn.Linear(D_in, H)  # input layer -> hidden layer 1
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(H, H)  # hidden layer 1 -> hidden layer 2
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(H, H)  # hidden layer 2 -> hidden layer 3
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(H, D_out)  # hidden layer 3 -> output layer
            self.relu4 = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x.float())  # convert input to float to avoid casting errors
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.fc4(x)
            return self.relu4(x)

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
