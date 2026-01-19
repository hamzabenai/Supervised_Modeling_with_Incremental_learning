from collections import deque
import random

def _new_deque(maxlen):
    return deque(maxlen=maxlen)

class ReplayBuffer:
    def __init__(self, max_per_class=100, seed=42):
        self.max_per_class = max_per_class
        self.buffer = {}
        self.seed = seed
        random.seed(seed)

    def _ensure_class(self, y):
        if y not in self.buffer:
            self.buffer[y] = deque(maxlen=self.max_per_class)

    def add(self, x, y):
        self._ensure_class(y)
        self.buffer[y].append(x)

    def replay(self):
        samples = []
        for y, xs in self.buffer.items():
            for x in xs:
                samples.append((x, y))
        random.shuffle(samples)
        return samples

    def is_empty(self):
        return len(self.buffer) == 0
