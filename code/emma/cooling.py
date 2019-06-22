"""
    A set of schedulers controlling the temperature decrease in simulated 
    annealing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod


# --------------- 
class Scheduler(ABC):
    """
        Base class.
        
        Arguments
            - temperature: starting temperature
            - alpha: rate of decrease
    """

    def __init__(self, temperature, alpha=0):
        self.T = temperature
        self.alpha = alpha
        self.init_T = self.T
        self.step = 0
        super().__init__()

    @property
    def temperature(self):
        return self.T
    
    @abstractmethod
    def cooldown(self):
        pass


# --------------- 
class ExponentialScheduler(Scheduler):
    def cooldown(self):
        self.T = self.init_T * pow(self.alpha, self.step)
        self.step += 1
        return self


# --------------- 
class LogarithmicScheduler(Scheduler):
    def cooldown(self):
        self.T = self.init_T / (1 + self.alpha * np.log(1 + self.step))
        self.step += 1
        return self


# --------------- 
class LinearScheduler(Scheduler):
    def cooldown(self):
        self.T = self.init_T / (1 + self.alpha * self.step)
        self.step += 1
        return self


# --------------- 
class QuadraticScheduler(Scheduler):
    def cooldown(self):
        self.T = self.init_T / (1 + self.alpha * self.step**2)
        self.step += 1
        return self


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    scheduler = QuadraticScheduler(10, 0.1)
    t = []
    for i in range(100):
        t.append(scheduler.cooldown().temperature)
    plt.plot(t)
    plt.show()

    







