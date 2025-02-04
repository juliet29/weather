def general_bezier(t:float, Bs:list[float]):
    # we have n+1 control points.. 
    # and range function goes to n-1
    n = len(Bs) - 1
    def inner_eq(i):
        return comb(n,i) * (1-t)**(n-i) * t**i * Bs[i]
    return sum(inner_eq(i) for i in range(n+1))

def eq(t, b1, b2):
        return (1-t)*b1 + t*b2


def test_general_bezier_on_two_deg():
    bxs = [1,2]
    for t in np.arange(step=0.1, stop=1):
        res = eq(t, *bxs)
        gen = general_bezier(t, bxs)
        assert res == gen
        print(f"{t:.2}: [{res:.2}, {gen:.2}]")