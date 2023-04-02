# RainbowDQN

This is a RainbowDQN fork for GNNs on Hex graphs.

Key files include:
+ [train.py](train.py): The entrypoint for RainbowDQN. Handles the training and acting loop.
+ [common/rainbow.py](common/rainbow.py): Manages the neural network training according to the RainbowDQN algorithm.
+ [common/replay\_buffer.py](common/replay_buffer.py): A prioritized replay buffer to store transitions of RainbowDQN.
+ [common/argp.py](common/argp.py): Parameters and arguments
