import numpy as np
import gym


class ModifiedEnv(gym.ObservationWrapper):
    def __init__(
        self,
        name="halfcheetah",
        pomdp_type="remove_velocity",
        flicker_prob=0.2,
        random_noise_sigma=0.1,
    ):
        if name == "halfcheetah":
            orig = gym.make("HalfCheetah-v3")
            self.name = "HalfCheetah-"
        else:
            orig = gym.make("Ant-v3")
            self.name = "Ant-"
        super().__init__(orig)
        self.finite_actions = False
        self.action_space_n = self.action_space.shape[0]

        if pomdp_type == "remove_velocity":
            self.name += "vel"
        elif pomdp_type == "flickering":
            self.name += f"fprob{flicker_prob:.2f}"
        elif pomdp_type == "random_noise":
            self.name += f"rnoise{random_noise_sigma:.2f}"

        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob
        self.random_noise_sigma = random_noise_sigma

        if self.pomdp_type == "remove_velocity":
            self.remain_obs_idx, self.observation_space = self._remove_velocity(name)
        self.observation_space_n = self.observation_space.shape[0]

    def observation(self, obs):
        if self.pomdp_type == "remove_velocity":
            return obs.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == "flickering":
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(obs.shape)
            else:
                return obs.flatten()
        elif self.pomdp_type == "random_noise":
            return (
                obs + np.random.normal(0, self.random_noise_sigma, obs.shape)
            ).flatten()
        elif self.pomdp_type == "random_sensor_missing":
            obs[np.random.rand(len(obs)) <= self.random_sensor_missing_prob] = 0
            return obs.flatten()

    @staticmethod
    def _remove_velocity(name):
        if name == "halfcheetah":
            remain_obs_idx = np.arange(0, 8)
        else:
            remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
        obs_low = np.array(
            [-np.inf for _ in range(len(remain_obs_idx))], dtype="float32"
        )
        obs_high = np.array(
            [np.inf for _ in range(len(remain_obs_idx))], dtype="float32"
        )
        observation_space = gym.spaces.Box(obs_low, obs_high)
        return remain_obs_idx, observation_space
