def prune(t):
    if t.label != None:
        return
    L = t.leaf_sum
    N = t.N
    p = (t.RT + 0.5*L)/N
    if t.RT + 0.5 - (N*p*(1-p))**0.5 < t.RT + 0.5*L:
        t.label = t.majority
        return
    prune(t.left)
    prune(t.right)