import scipy.stats as stat


def calculate_EMD(x: list, y: list) -> float:
    return stat.wasserstein_distance(x, y)
