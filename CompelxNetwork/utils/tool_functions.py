def print_progress(now: int, total: int, length: int = 40, prefix: str = 'progress:'):
    """
    print with a process bar.

    Parameters
    ----------
    now : process
    total : end of loops
    length : length of print bar
    prefix : the prefix print title

    """
    print('\r' + prefix + ' %.2f%%\t' % (now / total * 100), end='')
    print('[' + '>' * int(now / total * length) + '-' * int(length - now / total * length) + ']', end='')
