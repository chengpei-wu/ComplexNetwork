from copy import deepcopy


def havel_hakimi_process(degrees: list, p=False) -> list:
    """
    make a sequence to be a valid degree sequence for constructing a simple graph, based on Havel Hakimi algorithm,

    Parameters
    ----------
    degrees : the original degree sequence
    p: print degrees or not

    Returns
    -------
    the valid degree sequence

    """
    degrees.sort(reverse=True)
    ori_degrees = deepcopy(degrees)
    if ori_degrees[0] > len(ori_degrees) - 1:
        ori_degrees[0] = len(ori_degrees) - 1
    cnt = 0
    while degrees:
        if p:
            print(degrees)
        d = degrees[0]
        degrees = degrees[1:]
        if d > len(degrees):
            ori_degrees[cnt] = ori_degrees[cnt] - (len(degrees) - d)
        if d < 0:
            ori_degrees[cnt] = ori_degrees[cnt] + abs(d)

        minus = 0
        for i in range(min(d, len(degrees))):
            minus += 1
            degrees[i] -= 1
        if minus < d:
            ori_degrees[cnt] = ori_degrees[cnt] - (d - minus)
        cnt += 1
    return ori_degrees
