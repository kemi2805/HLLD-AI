"""numpy vs numba elementary math on this node: how many of 1e6 values differ."""
import math
import numpy as np
from numba import njit

np.show_runtime()
rng = np.random.default_rng(0)
x = rng.uniform(0.3, 3.0, 1_000_000)


@njit(cache=False)
def nb_pow3(a):
    o = np.empty_like(a)
    for i in range(a.size):
        o[i] = a[i] ** 3.0
    return o


@njit(cache=False)
def nb_pow4(a):
    o = np.empty_like(a)
    for i in range(a.size):
        o[i] = a[i] ** 4.0
    return o


@njit(cache=False)
def nb_f(a, k):
    o = np.empty_like(a)
    for i in range(a.size):
        v = a[i]
        o[i] = math.sqrt(v) if k == 0 else (math.exp(v) if k == 1 else math.log(v))
    return o


for name, ref, got in (("pow(x, 3.0)", np.power(x, 3.0), nb_pow3(x)),
                       ("pow(x, 4.0)", np.power(x, 4.0), nb_pow4(x)),
                       ("x ** 3.0 (numpy op)", x ** 3.0, nb_pow3(x)),
                       ("sqrt", np.sqrt(x), nb_f(x, 0)),
                       ("exp", np.exp(x), nb_f(x, 1)),
                       ("log", np.log(x), nb_f(x, 2))):
    d = ref != got
    print("%-22s differ %7d of %d  max rel %.1e" % (name, int(d.sum()), x.size,
          float(np.max(np.abs(ref - got) / np.abs(ref)))))
