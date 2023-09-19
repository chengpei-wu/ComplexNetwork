import scipy.stats


def calculate_EMD(x, y):
    return scipy.stats.wasserstein_distance(x, y)
