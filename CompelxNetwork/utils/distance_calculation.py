import scipy.stats


def calculate_EMD(x: list, y: list) -> float:
    return scipy.stats.wasserstein_distance(x, y)
