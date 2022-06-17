import numpy as np
from typing import Optional

from gym.envs.classic_control.cartpole import CartPoleEnv


class CartpoleEnv(CartPoleEnv):
    def __init__(
        self,
        pomdp_type="remove_velocity",
        flicker_prob=0.2,
    ):
        super(CartpoleEnv, self).__init__()
        self.name = "Cartpole-"
        if pomdp_type == "remove_velocity":
            self.name += "vel"
            self.observation_space_n = 2
        elif pomdp_type == "flickering":
            self.name += f"fprob{flicker_prob:.2f}"
            self.observation_space_n = 4
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.finite_actions = True
        self.action_space_n = 2
        self.max_ep_len = 200

    def step(self, action):
        state, reward, done, info = super(CartpoleEnv, self).step(action)
        if self.pomdp_type == "remove_velocity":
            partial_state = state[[0, 2]]
        else:
            partial_state = [0] * 4
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
        if self.pomdp_type == "remove_velocity":
            partial_state = self.state[[0, 2]]
        else:
            partial_state = [0] * 4
        self.steps_beyond_done = None
        if not return_info:
            return np.array(partial_state, dtype=np.float32)
        else:
            return np.array(partial_state, dtype=np.float32), {}
