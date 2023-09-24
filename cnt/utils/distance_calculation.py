import scipy.stats as stat


def calculate_EMD(x: list, y: list) -> float:
    """
    calculate the wasserstein distance (Earth Mover Distance, EMD) of two distributions.
    """
    return stat.wasserstein_distance(x, y)
