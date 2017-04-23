import math


class LinearDecay:
    def __init__(self, minimum, slope):
        self.minimum = minimum
        self.slope = slope

    def update(self, time_step):
        value = self.slope * time_step + 1.0
        return max(value, self.minimum)


class ExponentialDecay:
    def __init__(self, minimum, decay_rate):
        self.minimum = minimum
        self.decay_rate = decay_rate

    def update(self, time_step):
        value = math.exp(-self.decay_rate * time_step)
        return max(value, self.minimum)
