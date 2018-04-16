import numpy as np

import gym

class ObservationSpace(gym.Space):
    def __init__(self):
        super(ObservationSpace, self).__init__()

    def sample(self):
        super(ObservationSpace, self).sample()

    def contains(self):
        super(ObservationSpace, self).contains()

    def to_jsonable(self, sample_n):
        super(ObservationSpace, self).to_jsonable()

    def from_jsonable(self, sample_n):
        super(ObservationSpace, self).from_jsonable()
