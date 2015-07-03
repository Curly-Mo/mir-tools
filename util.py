from itertools import izip
from collections import Counter


def grouper(iterable, n, padvalue=None):
    """
    Group interable into chunks of size n
    Removes last group if shorter than n
    grouper('abcdefg', 3) --> (('a','b','c'), ('d','e','f'))
    """
    return izip(*[iter(iterable)]*n)


def most_common(values):
    most_common = Counter(values).most_common(1)[0][0]
    return most_common
