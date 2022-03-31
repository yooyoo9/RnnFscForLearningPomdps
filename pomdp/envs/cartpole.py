import numpy as np
from typing import Optional

from gym.envs.classic_control.cartpole import CartPoleEnv


class CartpoleEnv(CartPoleEnv):
    def __init__(self):
        super(CartpoleEnv, self).__init__()
        self.name = 'cartpole'
        self.observation_space_n = 2
        self.action_space_n = 2
        self.max_ep_len = 200

    def step(self, action):
        state, reward, done, info = super(CartpoleEnv, self).step(action)
        partial_state = state[[0, 2]]
        return np.array(partial_state, dtype=np.float32), reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        partial_state = self.state[[0, 2]]
        self.steps_beyond_done = None
        if not return_info:
            return np.array(partial_state, dtype=np.float32)
        else:
            return np.array(partial_state, dtype=np.float32), {}

