import numpy as np
import scipy
import scipy.stats as stats

class THSimulation():
    def __init__(self, nb_bandits, p_bandits, n=100000):
        self.nb_bandits = nb_bandits
        self.p_bandits = p_bandits
        self.n = n
        self.trials = [0] * self.nb_bandits
        self.wins = [0] * self.nb_bandits
    def pull(self, i):
        if np.random.rand() < self.p_bandits[i]:
            return 1
        else:
            return 0
    def step(self):
        # Define the prior based on current observations
        bandit_priors = [stats.beta(a=1+w, b=1+t-w) for t, w in zip(self.trials, self.wins)]
        # Sample a probability theta for each bandit
        theta_samples = [d.rvs(1) for d in bandit_priors]
        # choose a bandit
        chosen_bandit = np.argmax(theta_samples)
        # Pull the bandit
        x = self.pull(chosen_bandit)
        # Update trials and wins (defines the posterior)
        self.trials[chosen_bandit] += 1
        self.wins[chosen_bandit] += x
        return self.trials, self.wins
    
class THSimulationAdv():
    def __init__(self, nb_bandits):
        self.nb_bandits = nb_bandits
        self.trials = [0] * self.nb_bandits
        self.wins = [0] * self.nb_bandits
    def pull(self, i, p_bandits):
        if np.random.rand() < p_bandits[i]:
            return 1
        else:
            return 0
    def step(self, p_bandits):
        # Define the prior based on current observations
        bandit_priors = [stats.beta(a=1+w, b=1+t-w) for t, w in zip(self.trials, self.wins)]
        # Sample a probability theta for each bandit
        theta_samples = [d.rvs(1) for d in bandit_priors]
        # choose a bandit
        chosen_bandit = np.argmax(theta_samples)
        # Pull the bandit
        x = self.pull(chosen_bandit, p_bandits)
        # Update trials and wins (defines the posterior)
        self.trials[chosen_bandit] += 1
        self.wins[chosen_bandit] += x
        return self.trials, self.wins
