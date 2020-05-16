# This is Pessimistic Error Pruning.
def prune(t):
    # If the node is a leaf node, stop pruning.
    if t.label != None:
        return
    L = t.leaf_sum
    N = t.N
    p = (t.RT + 0.5*L)/N
    # When the following condition is satisfied, 
    # replace the subtree with a leaf node.
    if t.RT + 0.5 - (N*p*(1-p))**0.5 < t.RT + 0.5*L:
        # The label of the leaf node depends on the 
        # majority label of the subtree.
        t.label = t.majority
        return
    # Prune the tree recursively.
    prune(t.left)
    prune(t.right)