# Multi-agent Path Finding for Cooperative Autonomous Driving

This is the code for the paper [Multi-agent Path Finding for Cooperative Autonomous Driving
](https://arxiv.org/abs/2402.00334) published at ICRA 2024.

```
@inproceedings{yan2024multi,
title={Multi-agent Path Finding for Cooperative Autonomous Driving},
author={Zhongxia Yan and Han Zheng and Cathy Wu},
booktitle={International Conference of Robotics and Automation (ICRA)},
year={2024},
}
```

## Setup

Apologies for the sparse instructions. Clone this repo and add it to the `PYTHONPATH`. Install packages as needed.

## Run
Again sorry for the sparse instructions.

The main script is `bicycle_model.py`, and it can be parameterized to run with different search methods and intersection settings.

For FIFO: run the script as is.

For OBS: use `method='obs'` and set `obs_order_threshold` parameter to control number of OBS orders searched.

For PP: use `method='pp'` and set `num_random_orders` to control number of PP orders.

For MCTS: use `method='mcts'` and set `num_mcts_simulations` to control number of MCTS simulations.