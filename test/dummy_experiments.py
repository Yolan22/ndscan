import numpy as np

class SineWaveExperiment:
    def __init__(self, d_dimensions=3, m_shots=5, noise_level=2.0):
        self.d = d_dimensions
        self.m_shots = m_shots
        self.noise_level = noise_level  # standard deviation of the environmental noise
        
        np.random.seed(42)
        self.true_weights = np.random.uniform(1.0, 5.0, size=self.d)
        self.x_base = np.linspace(0, 10, 100)
        
        # ground truth signal
        self.true_signal = np.zeros_like(self.x_base)
        for i, w in enumerate(self.true_weights):
            self.true_signal += w * np.sin((i + 1) * self.x_base)

    def evaluate(self, X_weights):
        """
        Expects an input array X of shape (n, d).
        Simulates m_shots per sample by adding random noise to each shot.
        
        Outputs:
            - means: array of shape (n,) -> The average score across the m-shots.
            - stds:  array of shape (n,) -> The standard deviation of the mean (Standard Error).
        """
        X_weights = np.atleast_2d(X_weights)
        n, d = X_weights.shape
        
        means = []
        stds = []
        
        for candidate in X_weights:
            # generate noisless underlying signal for this candidate
            candidate_signal = np.zeros_like(self.x_base)
            for i, w in enumerate(candidate):
                candidate_signal += w * np.sin((i + 1) * self.x_base)
            
            # find true MSE score (negative because we want to maximize the score)
            true_mse = np.mean((self.true_signal - candidate_signal) ** 2)
            base_score = -true_mse
            
            # simulate m independent shots by adding random gaussian noise to each shot
            shot_scores = []
            for _ in range(self.m_shots):
                # add random variations to the score for each shot
                noisy_score = base_score + np.random.normal(0, self.noise_level)
                shot_scores.append(noisy_score)
            
            shot_scores = np.array(shot_scores)
            
            # calculate empirical mean across shots
            means.append(np.mean(shot_scores))
            
            # calculate the Standard Error of the Mean (SEM)
            # This represents the uncertainty of our sample average
            sem = np.std(shot_scores, ddof=1) / np.sqrt(self.m_shots)
            stds.append(sem)
            
        return np.array(means), np.array(stds)
    
class BraninMaximizationExperiment:
    """
    Global Maxima: Score ≈ -0.397887 
    At points: (-pi, 12.275), (pi, 2.275), and (9.42478, 2.475)
    Bounds usually: x1 in [-5, 10], x2 in [0, 15]
    """
    def __init__(self, m_shots=5, noise_level=0.1):
        self.d = 2  # Branin is strictly 2D
        self.m_shots = m_shots
        self.noise_level = noise_level

    def evaluate(self, X):
        X = np.atleast_2d(X)
        means, stds = [], []
        
        # Standard constants for Branin
        a = 1.0
        b = 5.1 / (4 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * np.pi) # Note: 1/(8*pi) aligns with standard global minima of 0.397887.
        
        for candidate in X:
            x1, x2 = candidate[0], candidate[1]
            
            # Standard formula (minimization variant)
            f = a * (x2 - b*x1**2 + c*x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
            base_score = -f  # Invert for maximization
            
            shot_scores = base_score + np.random.normal(0, self.noise_level, size=self.m_shots)
            means.append(np.mean(shot_scores))
            stds.append(np.std(shot_scores, ddof=1) )
            
        return np.array(means), np.array(stds)
    
class RosenbrockMaximizationExperiment:
    """
    Global Maximum: Score = 0.0
    At point: [1.0, 1.0, ..., 1.0]
    Bounds usually: x_i in [-2.0, 2.0]
    """
    def __init__(self, d_dimensions=3, m_shots=5, noise_level=0.1):
        self.d = d_dimensions
        self.m_shots = m_shots
        self.noise_level = noise_level

    def evaluate(self, X):
        X = np.atleast_2d(X)
        means, stds = [], []
        
        for candidate in X:
            # Rosenbrock summation logic
            f = 0.0
            for i in range(self.d - 1):
                f += 100 * (candidate[i+1] - candidate[i]**2)**2 + (1 - candidate[i])**2
            
            base_score = -f  # Invert for maximization
            
            shot_scores = base_score + np.random.normal(0, self.noise_level, size=self.m_shots)
            means.append(np.mean(shot_scores))
            stds.append(np.std(shot_scores, ddof=1))
            
        return np.array(means), np.array(stds)

class AckleyMaximizationExperiment:
    """
    Global Maximum: Score = 0.0
    At point: [0.0, 0.0, ..., 0.0]
    Bounds usually: x_i in [-32.768, 32.768]
    """
    def __init__(self, d_dimensions=3, m_shots=5, noise_level=0.1):
        self.d = d_dimensions
        self.m_shots = m_shots
        self.noise_level = noise_level

    def evaluate(self, X):
        X = np.atleast_2d(X)
        means, stds = [], []
        
        for candidate in X:
            sum_sq = np.sum(candidate**2)
            sum_cos = np.sum(np.cos(2 * np.pi * candidate))
            
            term1 = -0.2 * np.sqrt(sum_sq / self.d)
            term2 = sum_cos / self.d
            
            f = 20 * np.exp(term1) + np.exp(term2) - 20 - np.e
            
            base_score = -f
            
            shot_scores = base_score + np.random.normal(0, self.noise_level, size=self.m_shots)
            means.append(np.mean(shot_scores))
            stds.append(np.std(shot_scores, ddof=1))
            
        return np.array(means), np.array(stds)
    
class RastriginMaximizationExperiment:
    """
    Global Maximum: Score = 0.0
    At point: [0.0, 0.0, ..., 0.0]
    Bounds usually: x_i in [-5.12, 5.12]
    """
    def __init__(self, d_dimensions=3, m_shots=5, noise_level=0.1):
        self.d = d_dimensions
        self.m_shots = m_shots
        self.noise_level = noise_level

    def evaluate(self, X):
        X = np.atleast_2d(X)
        means, stds = [], []
        
        for candidate in X:
            f = 10 * self.d + np.sum(candidate**2 - 10 * np.cos(2 * np.pi * candidate))
            base_score = -f  # invert for maximization
            
            shot_scores = base_score + np.random.normal(0, self.noise_level, size=self.m_shots)
            means.append(np.mean(shot_scores))
            stds.append(np.std(shot_scores, ddof=1))
            
        return np.array(means), np.array(stds)

class SimpleExperiment():
    """ A simple experiment to test the optimization framework.
    
    Global Minimum : Score = 0.0
    At point: [1.25, 1.25, 1.25]
    Bounds usually: x_i in [-2.0, 2.0]
    """
    def __init__(self, m_shots=5, noise_level=0.1):
        self.d = 3
        self.m_shots = m_shots
        self.noise_level = noise_level

    def evaluate(self, X):
        X = np.atleast_2d(X)
        means, stds = [], []
        
        for candidate in X:
            # true scalar objective
            f = np.sum((candidate - 1.25) ** 2) 
            base_score = -f # invert for maximization
            
            # noisy repeated measurements
            shot_scores = (base_score + np.random.normal(0,self.noise_level,size=self.m_shots))
            means.append(np.mean(shot_scores))
            stds.append(np.std(shot_scores, ddof=1))

        return np.array(means), np.array(stds)