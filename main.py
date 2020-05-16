import data
import ccp_prune
import pep_prune

shitseed = [0,0]

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


features, X = data.get_data('train.csv')
best_tree = ccp_prune.process(features,X, 1080)
features, X = data.get_data('test.csv')
confusion_matrix = [[0, 0], [0, 0]]
print('Test Accuracy = {}'.format(classification(best_tree.root, X)/len(X)))
print('Confusion Matrix = {}'.format(confusion_matrix))
