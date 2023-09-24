def uniform_sampling(input_curve: list, percentage_interval: int) -> list:
    """
    uniform sampling

    Parameters
    ----------
    input_curve : the original curve
    percentage_interval : the interval size

    Returns
    -------
    the sampled curve

    """

    if percentage_interval <= 0 or percentage_interval >= 100:
        raise ValueError('percentage_interval: must be 0~100')

    length = len(input_curve)
    sampled_length = round(100 / percentage_interval) + 1
    if sampled_length == length:
        return input_curve
    sampled_curve = [input_curve[round(length * (i / 100))] for i in
                     range(0, 100, percentage_interval)]
    sampled_curve.append(input_curve[-1])
    return sampled_curve
