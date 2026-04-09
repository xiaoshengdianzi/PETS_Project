import collections
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done):
        # Debug打印，帮助定位shape和类型问题
        state = np.copy(np.asarray(state))
        action = np.copy(np.asarray(action))
        reward = np.copy(np.asarray(reward))
        next_state = np.copy(np.asarray(next_state))
        done = np.copy(np.asarray(done))
        # 检查shape一致性
        if len(self.buffer) > 0:
            s0, a0, r0, ns0, d0 = self.buffer[0]
            if state.shape != s0.shape:
                raise ValueError(f"State shape mismatch: {state.shape} vs {s0.shape}")
            if action.shape != a0.shape:
                raise ValueError(f"Action shape mismatch: {action.shape} vs {a0.shape}")
            if reward.shape != r0.shape:
                raise ValueError(f"Reward shape mismatch: {reward.shape} vs {r0.shape}")
            if next_state.shape != ns0.shape:
                raise ValueError(f"Next_state shape mismatch: {next_state.shape} vs {ns0.shape}")
            if done.shape != d0.shape:
                raise ValueError(f"Done shape mismatch: {done.shape} vs {d0.shape}")
        self.buffer.append((state, action, reward, next_state, done))
    def size(self):
        return len(self.buffer)
    def return_all_samples(self):
        all_transitions = list(self.buffer)
        state, action, reward, next_state, done = zip(*all_transitions)
        try:
            return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)
        except ValueError:
            # fallback: shape不一致时返回object数组，便于调试
            return (
                np.array(state, dtype=object),
                np.array(action, dtype=object),
                np.array(reward, dtype=object),
                np.array(next_state, dtype=object),
                np.array(done, dtype=object)
            )
