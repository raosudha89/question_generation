import random
def foo(y):
    random.shuffle(y)
    return 1. / (1. + y.index(0))

x = range(10)
#N = 1000000
print sum((foo(x) for _ in xrange(10000))) / 10000.
