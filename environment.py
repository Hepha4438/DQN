""" Environment wrapper for Atari, following Nature DQN (Mnih et al., 2015) """
import gymnasium as gym
import ale_py
import numpy as np
import random
from skimage import color, transform
import warnings


class Environment(object):
    def __init__(self, name="ALE/Breakout-v5", width=84, height=84, history=4, frame_skip=4):
        self._env = gym.make(name, render_mode="rgb_array")
        obs, info = self._env.reset()

        self._width = width
        self._height = height
        self._history = history
        self._frame_skip = frame_skip

        self._reward = 0
        self._terminal = True
        self._screen = None
        self._screen_ori = None

        # Action mapping
        self._action_meanings = self._env.unwrapped.get_action_meanings()
        self._noop_action = self._action_meanings.index("NOOP") if "NOOP" in self._action_meanings else 0
        self._fire_action = self._action_meanings.index("FIRE") if "FIRE" in self._action_meanings else None

    @property
    def action_size(self):
        return self._env.action_space.n

    @property
    def state(self):
        return self._screen, self._reward, self._terminal

    @property
    def screen(self):
        return self._screen_ori

    @property
    def lives(self):
        if hasattr(self._env, "ale"):
            return self._env.ale.lives()
        warnings.warn("Env has no ALE interface, returning lives=0", RuntimeWarning)
        return 0

    def new_random_game(self, force=False):
        if self.lives == 0 or force:
            obs, info = self._env.reset()
            self._screen_ori = obs
            self._screen = transform.resize(obs, [self._height, self._width])
            self._screen = color.rgb2gray(self._screen)
            self._terminal = False

        noops = random.randint(1, 30)
        for _ in range(noops):
            self._step(self._noop_action)
            if self._terminal:
                obs, info = self._env.reset()
                self._screen_ori = obs
                self._screen = transform.resize(obs, [self._height, self._width])
                self._screen = color.rgb2gray(self._screen)
                self._terminal = False

        # FIRE to start ball if required (Breakout)
        if self._fire_action is not None:
            self._step(self._fire_action)
            if self._terminal:
                obs, info = self._env.reset()
                self._screen_ori = obs
                self._screen = transform.resize(obs, [self._height, self._width])
                self._screen = color.rgb2gray(self._screen)
                self._terminal = False

        return self._screen, 0, 0, self._terminal

    def act(self, action, is_train=True):
        start_lives = self.lives
        self._step(action)
        if is_train and start_lives > self.lives:
            self._reward -= 1.0
            self._terminal = True
        return self.state

    def _step(self, action):
        total_reward = 0.0
        frames = []
        terminated, truncated = False, False

        for t in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            frames.append(obs)
            if terminated or truncated:
                break

        # Max-pool over last two frames
        if len(frames) >= 2:
            obs = np.maximum(frames[-2], frames[-1])
        else:
            obs = frames[-1]

        self._screen_ori = obs
        self._reward = total_reward
        self._terminal = terminated or truncated

        # Resize + grayscale
        self._screen = transform.resize(obs, [self._height, self._width])
        self._screen = color.rgb2gray(self._screen)


class ReplayMemory(object):
    def __init__(self, env, capacity, batch_size):
        self._history = env._history
        self._height = env._height
        self._width = env._width
        self._capacity = capacity
        self._batch_size = batch_size
        self._actions = np.empty(self._capacity, dtype=np.int32)
        self._rewards = np.empty(self._capacity, dtype=np.float32)
        self._screens = np.empty((self._capacity, self._height, self._width), dtype=np.float32)
        self._terminals = np.empty(self._capacity, dtype=bool)
        self._count = 0
        self._current = 0
        self._prestat = np.empty((self._batch_size, self._history, self._height, self._width), dtype=np.float32)
        self._poststat = np.empty((self._batch_size, self._history, self._height, self._width), dtype=np.float32)
        print("Replay memory initialized")

    def add(self, screen, reward, action, terminal):
        self._actions[self._current] = action
        self._rewards[self._current] = reward
        self._screens[self._current, ...] = screen
        self._terminals[self._current] = terminal
        self._count = max(self._count, self._current + 1)
        self._current = (self._current + 1) % self._capacity

    def sample(self):
        indexes = []
        while len(indexes) < self._batch_size:
            while True:
                index = random.randint(self._history, self._count - 1)
                if index >= self._current and index - self._history < self._current:
                    continue
                if self._terminals[(index - self._history):index].any():
                    continue
                break
            self._prestat[len(indexes), ...] = self._get_state(index - 1)
            self._poststat[len(indexes), ...] = self._get_state(index)
            indexes.append(index)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        terminals = self._terminals[indexes]
        return self._prestat, actions, rewards, self._poststat, terminals

    def _get_state(self, index):
        index = index % self._count
        if index >= self._history - 1:
            return self._screens[(index - (self._history - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self._count for i in reversed(range(self._history))]
            return self._screens[indexes, ...]


class History(object):
    def __init__(self, env):
        self._history = env._history
        self._height = env._height
        self._width = env._width
        self._input = np.zeros([self._history, self._height, self._width], dtype=np.float32)

    @property
    def get(self):
        return self._input

    @property
    def reset(self):
        self._input *= 0

    def add(self, screen):
        self._input[:-1] = self._input[1:]
        self._input[-1] = screen
