import numpy as np

from gym import Space

class ObservationSpace(Space):
    max_ratio = 3.0

    def __init__(self):
        super(ObservationSpace, self).__init__()

    def sample(self):
        return np.random.uniform(0, ObservationSpace.max_ratio, 4)

    def contains(self, obs):
        return len(obs) == 4 and (obs >= 0.0).all() and (x <= ObservationSpace.max_ratio).all()       

    def to_jsonable(self, sample_n):
        return np.array(sample_n).to_list()

    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]
