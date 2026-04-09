import numpy as np
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import EnsembleDynamicsModel
from cem import CEM
from fake_env import FakeEnv
from replay_buffer import ReplayBuffer

class PETS:
    ''' PETS 算法 '''
    def __init__(self, env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes):
        self._env = env
        self._env_pool = replay_buffer
        obs_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._model = EnsembleDynamicsModel(obs_dim, self._action_dim)
        self._fake_env = FakeEnv(self._model)
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self._cem = CEM(n_sequence, elite_ratio, self._fake_env, self.upper_bound, self.lower_bound)
        self.plan_horizon = plan_horizon
        self.num_episodes = num_episodes
    def train_model(self):
        env_samples = self._env_pool.return_all_samples()
        obs = env_samples[0]
        actions = np.array(env_samples[1])
        rewards = np.array(env_samples[2]).reshape(-1, 1)
        next_obs = env_samples[3]
        inputs = np.concatenate((obs, actions), axis=-1)
        labels = np.concatenate((rewards, next_obs - obs), axis=-1)
        self._model.train(inputs, labels)
    def mpc(self):
        mean = np.tile((self.upper_bound + self.lower_bound) / 2.0, self.plan_horizon)
        var = np.tile(np.square(self.upper_bound - self.lower_bound) / 16, self.plan_horizon)
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        done, episode_return = False, 0
        while not done:
            actions = self._cem.optimize(obs, mean, var)
            action = actions[:self._action_dim]
            step_result = self._env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
            mean = np.concatenate([np.copy(actions)[self._action_dim:], np.zeros(self._action_dim)])
        return episode_return
    def explore(self):
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        done, episode_return = False, 0
        while not done:
            action = self._env.action_space.sample()
            step_result = self._env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result
            self._env_pool.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_return += reward
        return episode_return
    def train(self):
        return_list = []
        explore_return = self.explore()
        print('episode: 1, return: %d' % explore_return)
        return_list.append(explore_return)
        for i_episode in range(self.num_episodes-1):
            self.train_model()
            episode_return = self.mpc()
            return_list.append(episode_return)
            print('episode: %d, return: %d' % (i_episode+2, episode_return))
        return return_list

if __name__ == '__main__':
    buffer_size = 100000
    n_sequence = 50
    elite_ratio = 0.2
    plan_horizon = 25
    num_episodes = 10
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    replay_buffer = ReplayBuffer(buffer_size)
    pets = PETS(env, replay_buffer, n_sequence, elite_ratio, plan_horizon, num_episodes)
    return_list = pets.train()
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PETS on {}'.format(env_name))
    plt.savefig('result.png')
    plt.close()
