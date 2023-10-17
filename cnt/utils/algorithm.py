from copy import deepcopy


def havel_hakimi_process(degrees: list) -> list:
    """
    make a sequence to be a valid degree sequence for constructing a simple graph, based on Havel Hakimi algorithm,

    Parameters
    ----------
    degrees : the original degree sequence

    Returns
    -------
    the valid degree sequence

    """
    degrees.sort(reverse=True)
    ori_degrees = deepcopy(degrees)
    cnt = 0
    while degrees:
        # print(degrees)
        d = degrees[0]
        degrees = degrees[1:]
        if d > len(degrees):
            ori_degrees[cnt] = ori_degrees[cnt] - (len(degrees) - d)
        if d < 0:
            ori_degrees[cnt] = ori_degrees[cnt] + abs(d)
        cnt += 1
        for i in range(min(d, len(degrees))):
            degrees[i] -= 1

    return ori_degrees
