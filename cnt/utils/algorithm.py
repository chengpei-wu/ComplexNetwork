from copy import deepcopy

import numpy as np


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
    degrees = np.array(degrees)
    ori_degrees = deepcopy(degrees)
    cnt = 0
    while len(degrees) > 0:
        if p:
            print('de:', degrees)
        d = degrees[0]
        degrees = degrees[1:]

        if d > len(degrees) > 0:
            ori_degrees[cnt] = len(degrees)

        if d < 0 < len(degrees):
            ori_degrees[cnt] = ori_degrees[cnt] + abs(d)

        if len(degrees) == 0 and d != 0:
            if p:
                print('**', ori_degrees, d)
            ori_degrees[cnt] -= d
            print(ori_degrees)
            break

        minus = 0
        for i in range(min(d, len(degrees))):
            minus += 1
            degrees[i] -= 1
        if minus < d and len(degrees) != 0:
            ori_degrees[cnt] = ori_degrees[cnt] - (d - minus)
        cnt += 1
    return list(ori_degrees)
