import sympy
import math
import numpy as np


def tuple_gen(order, var_n):
    if var_n == 0:
        return [[order]]
    l = []
    for i in range(order + 1):
        r = tuple_gen(order-i, var_n - 1)
        l += [[i]+t for t in r]
    l = sorted(l, key=lambda x: (-sum(i**2 for i in x), x[::-1]))
    return l

def bernstein_evaluator(x, y, z, codecs):
    m = len(codecs[0]) # dim + 1
    n = codecs[0][0] # order
    mc_dict = sympy.multinomial_coefficients(m, n)
    mc = np.array([mc_dict[tuple(c)] for c in codecs])

    w = 1-x-y-z
    computed_powers = np.array([(w**i, x**i, y**i, z**i)
                              for i in range(n + 1)])  # make use of 0**0 == 1
    return mc[:,None]*np.array(
        [np.prod([computed_powers[c, i] for i, c in enumerate(cod)], axis=0) for cod in codecs])

