from copy import deepcopy


def gini(li):
    n = len(li)
    m = sum([i[-1] for i in li])
    return m * (n - m) / n / n


def count_majority(li):
    return 2 * sum([i[-1] for i in li]) > len(li)


class Node(object):
    def __init__(self, id=None, name=None, key=None, left=None, right=None, label=None, leaf_sum=0, Rt=None, RT=None, majority=None):
        self.feature_id = id
        self.feature_name = name

        self.split_value = key

        self.left = left
        self.right = right

        self.leaf_sum = leaf_sum

        self.label = label
        self.Rt = Rt
        self.RT = RT

        self.majority = majority


class Tree(object):
    def __init__(self, features):
        self.features = features
        self.root = None

    def build(self, X, dep, max_dep, N):
        def unique(li):
            for i in li:
                if i[-1] != li[0][-1]:
                    return False
            return True
        if unique(X) or dep == max_dep:
            return Node(label=X[0][-1], leaf_sum=1, Rt=0, RT=0)
        min_feature = None
        min_key = None
        min_Imp = 0
        min_id = None
        left = None
        right = None
        for i, name in enumerate(self.features):
            X.sort(key=lambda x: x[i])
            n = len(X)
            for j in range(n - 1):
                delta = gini(X) - (j + 1)/n * \
                    gini(X[:j+1]) - (n - j - 1)/n * gini(X[j+1:])
                if delta > min_Imp and X[j][i] != X[j+1][i]:
                    min_Imp = delta
                    min_feature = name
                    min_id = i
                    min_key = (X[j][i] + X[j+1][i]) / 2
                    left = X[:j+1]
                    right = X[j+1:]
        left = self.build(left, dep+1, max_dep, N)
        right = self.build(right, dep+1, max_dep, N)
        return Node(min_id, min_feature, min_key, left, right, None, left.leaf_sum+right.leaf_sum, gini(X)*len(X)*len(X)/N, left.RT+right.RT, count_majority(X))


def prune(t, alpha):
    if t.label != None:
        return
    if t.RT + alpha * t.leaf_sum >= t.Rt + alpha:
        t.label = t.majority
        return
    prune(t.left, alpha)
    prune(t.right, alpha)


def tree_size(t):
    if t.label != None:
        return 1
    return tree_size(t.left) + tree_size(t.right)


def cost_complexity_algorithm(tree, X):
    alphas = {0}

    def travel(t):
        if t.label != None:
            return
        alphas.add((t.Rt - t.RT)/(t.leaf_sum-1))
        travel(t.left)
        travel(t.right)
    travel(tree.root)
    alphas = sorted(alphas)
    pruned_trees = []
    for a in alphas:
        new_tree = deepcopy(tree)
        prune(new_tree.root, a)
        pruned_trees.append(new_tree)
    return pruned_trees


def shuffle(li, seed):
    for i in range(50):
        li.remove(li[seed])
        li.append(li[seed])


def process(features, X, seed):
    confusion_matrix = [[0, 0], [0, 0]]

    def classification(t, test_data):
        if t.label != None:
            res = 0
            for i in test_data:
                res += i[-1] == t.label
                confusion_matrix[int(t.label)][int(i[-1])] += 1
            return res
        left = []
        right = []
        for i in test_data:
            if i[t.feature_id] < t.split_value:
                left.append(i)
            else:
                right.append(i)
        return classification(t.left, left) + classification(t.right, right)

    best_tree = None
    best_score = 100000000
    for k in range(0,100,10):
        shuffle(X, seed + k)
        mid = int(len(X) * 0.5)

        my_tree = Tree(features)

        my_tree.root = my_tree.build(
            X[:mid], 0, 10, len(X[:mid]))

        pruned_trees = cost_complexity_algorithm(my_tree, X[:mid])

        # best_tree = pruned_trees[0]
        # best_score = classification(best_tree.root, X[mid:]) / len(X[mid:])
        for t in pruned_trees[1:]:
            confusion_matrix = [[0, 0], [0, 0]]
            score = classification(t.root, X[mid:]) / len(X[mid:])
            # cishu = classification(t.root, X[mid:])
            # if confusion_matrix[1][1] == 0:
                # confusion_matrix[1][1] = 0.0001
            # score = (confusion_matrix[1][0] + confusion_matrix[0]
                    #  [1])/(confusion_matrix[1][1]**2)/(cishu**0.3)
            if score < best_score:
                best_tree = t
                best_score = score
        
        return best_tree
