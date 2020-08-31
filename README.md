# DeepCubeA Exercises
These are exercises to understand the 
[DeepCubeA](https://www.ics.uci.edu/~fagostin/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf) 
algorithm.

These exercises are for anyone who is getting started with deep reinforcement learning.
The goal of these exercises is to implement a method that learns to solve the 8-puzzle.
The solutions to these exercises can be run on a standard laptop CPU in less than 10 minutes.
Sample outputs of solutions to each exercise are also provided.

This currently contains one exercise. More to come.

For any issues, please contact Forest Agostinelli (fagostin@uci.edu)

# Setup
These exercises require Python3, PyTorch, and numpy.

# Exercise 1: Supervised Learning
We would like to build a DNN that learns to approximate the cost-to-go from any state of the 8-puzzle to the 
goal state (the configuration of the solution). This also corresponds to the minimum number of moves required to 
solve the state.

Luckily, there is an oracle that can tell us the cost-to-go for any state. All we have to do is design a DNN
architecture that can map any 8-puzzle state to its estimated cost-to-go.

To complete this exercise, you will have to implement:
- `get_nnet_model` in `to_implement/functions.py`
    - This method returns the pytorch model (torch.nn.Module) that maps any 8-puzzle state to its cost-to-go.
    - The dimensionality of the input will be (B x 81), where B is the batch size. This is because the 8-puzzle has 9 tiles, including the blank tile. 
    The representation given to the neural network is a one-hot representation for each tile. The dimensionality of the output will be (B x 1).
- `train_nnet` in `to_implement/functions.py`
    - This method trains the pytorch model