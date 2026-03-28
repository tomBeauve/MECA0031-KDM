import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import math


def generate_LHS(n_sample, n_dim, bounds):
    permut = np.zeros((n_sample, n_dim))

    for i in range(n_dim):
        permut[:, i] = np.random.permutation(n_sample)

    jitter = np.random.uniform(size=(n_sample, n_dim))

    sample_coord_norm = (permut + jitter) / n_sample

    sample_coord = np.zeros_like(sample_coord_norm)

    for i in range(n_dim):
        low, high = bounds[i]
        sample_coord[:, i] = low + sample_coord_norm[:, i] * (high - low)

    return sample_coord, sample_coord_norm


def phi_q_norm(samples, q=2):
    """Calculates the Morris-Mitchell criterion."""
    # Calculate all pair-wise Euclidean distances
    distances = pdist(samples)
    return np.sum(distances**(-q))**(1/q)


def generate_optimal_LHS(n_sample, n_dim, bounds, iterations=10000, q=2):
    best_phi = math.inf
    best_LH = None

    for i in range(iterations):
        candidate_phys, candidate_norm = generate_LHS(n_sample, n_dim, bounds)
        candidate_phi = phi_q_norm(candidate_norm, q=q)

        if candidate_phi < best_phi:
            best_phi = candidate_phi
            best_LH = candidate_phys

    return best_LH
