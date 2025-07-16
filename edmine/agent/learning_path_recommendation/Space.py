import numpy as np


# ============================
# 模拟 Gym 的简化 Space 接口
# ============================

class Discrete:
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(self.n)

    def contains(self, x: int) -> bool:
        return isinstance(x, int) and 0 <= x < self.n


class Box:
    def __init__(self, low: float, high: float, shape: tuple[int], dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self):
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, x) -> bool:
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
