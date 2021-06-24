import numpy as np
import pandas as pd

class Replayer(object):
    def __init__(self, config, prioritized=True, size_times=10):
        self.memory = pd.DataFrame()
        self.insample_size = config.batch_size
        self.memory_max = self.insample_size*size_times
        self.alpha = 0.7
        self.prioritized = prioritized
    
    def __len__(self):
        return self.memory.shape[0]
    
    def store(self, data):
        if self.memory.shape[0] + self.insample_size >= self.memory_max:
            if self.memory.shape[0] > 0:
                to_remove = np.random.choice(self.memory.shape[0], self.insample_size, replace=False)
                self.memory = self.memory.drop(to_remove)
        self.memory = pd.concat([self.memory, data], ignore_index=True)
        if self.prioritized:
            self.memory = self.memory.sort_values(by=['advantage_abs'] , ascending=False)
    
    def sample(self, size=None):
        if self.prioritized:
            self.prob = np.arange(self.memory.shape[0])+1
            self.prob = np.power(self.prob, -self.alpha)
            self.prob /= np.sum(self.prob)
            indices = np.random.choice(self.memory.shape[0], size=self.insample_size, p=self.prob)
        else:
            indices = np.random.choice(self.memory.shape[0], size=self.insample_size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)
