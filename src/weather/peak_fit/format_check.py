import numpy as np

def is_odd(n:int):
    assert int(n) == n # check its an integer
    return n % 2 == 1

def is_x_formatted_correcly(x:np.array):
    def middle_is_0():
        assert x[mid_ix] == 0, f"len_x = {len_x}, mid_ix={mid_ix}, x[mid_ix] = {x[mid_ix]}"

    def is_sorted():
        assert (np.sort(x) == x).all()

    def is_symmetric_about_0():
        for ix in range(1, side_length+1):
            pos = x[mid_ix + ix]
            neg = x[mid_ix - ix]
            assert pos*-1 == neg

    len_x = len(x)
    is_odd(len_x)

    side_length = len_x // 2
    mid_ix = side_length 

    middle_is_0()
    is_sorted()
    is_symmetric_about_0()

    return True

    
