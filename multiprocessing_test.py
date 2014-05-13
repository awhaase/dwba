from multiprocessing import Pool

def f(x):
    return x*x

def calc():
    p = Pool()
    p1 = p.apply_async(f, [1])
    p2 = p.apply_async(f, [2])
    p3 = p.apply_async(f, [3])
    print p1.get()
    print p2.get()
    print p3.get()