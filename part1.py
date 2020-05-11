import numpy as np 
import matplotlib as plt
from collections import Counter
from math import log
import sys
import time

class ListQueue:

    def __init__(self,capacity):
        self.__capacity = capacity
        self.__data = [None] * self.__capacity
        self.__size = 0
        self.__front = 0
        self.__end = 0

    def __len__(self):
        return self.__size

    def is_empty(self):
        return self.__size == 0

    def first(self):
        if self.is_empty():
            print('Queue is empty.')
        else:
            return self.__data[self.__front]

    def dequeue(self):
        if self.is_empty():
            print('Queue is empty.')
            return None
        answer = self.__data[self.__front]
        self.__data[self.__front] = None
        self.__front = (self.__front + 1) % self.__capacity
        self.__size -= 1
        return answer

    def enqueue(self,e):
        if self.__size == self.__capacity:
            print('The queue is full.')
            return None
        self.__data[self.__end] = e
        self.__end = (self.__end + 1) % self.__capacity
        self.__size += 1

    def __str__(self):
        return str(self.__data)

    def __repr__(self):
        return str(self)

class TreeNode():
    """树结点"""
    def __init__(self, feature_idx=None, feature_val=None, feature_name=None, node_val=None, child=None):
        """
        feature_idx:
            该结点对应的划分特征索引
        feature_val:
            划分特征对应的值 二叉树 所以这里的value应该是一个特定的值
        feature_name:
            划分特征名
        node_val:
            该结点存储的值，**只有叶结点才存储类别**
        child:
            子树
        """
        self._feature_idx = feature_idx
        self._feature_val = feature_val
        self._feature_name = feature_name
        # 叶结点存储类别
        self._node_val = node_val
        # 非叶结点存储划分信息
        self._child = child
    def DFSearch(self):
        if self._child != None:
            print(self._feature_name)
            print(self._feature_val)
        if self._child is None:
            print(self._node_val)
            return
        else:
            if self._child[0] is not None:
                self._child[0].DFSearch()
            if self._child[1] is not None:
                self._child[1].DFSearch()
    
    def BFSearch(self):

        q = ListQueue(2000)
        q.enqueue(self)

        while q.is_empty() is False:
            cNode = q.dequeue()
            if cNode._child is not None:
                q.enqueue(cNode._child[0])
                q.enqueue(cNode._child[1])
            if cNode._feature_name is not None and cNode._feature_val is not None:
                print(cNode._feature_name)
                print(cNode._feature_val)
            elif cNode._node_val is not None:
                print(cNode._node_val)

class DecisionTreeScratch():
    """决策树算法Scratch实现"""
    def __init__(self, feature_name, etype="gain"):
        """
        feature_name:
            每列特征名
        etype:
            可取值有
            gain: 使用信息增益
            ratio: 使用信息增益比
            gini: 使用基尼系数
        """
        self._root = None
        self._fea_name = feature_name
        self._etype = etype


    def _build_tree(self, X, y):
        """
        构建树
        X:
            用于构建子树的数据集
        y:
            X对应的标签
        """
        # 子树只剩下一个类别时直接置为叶结点
        if np.unique(y).shape[0] == 1:
            return TreeNode(node_val=y[0])
        max_gain, max_fea_idx, fea_val = self.try_split(X,y,choice=self._etype)
        feature_name = self._fea_name[max_fea_idx]
        child_tree = dict()
        # 遍历所选特征每一个可能的值，对每一个值构建子树
        # 该子树对应的数据集和标签
        child_X_l = X[X[:, max_fea_idx] <= fea_val]
        child_y_l = y[X[:, max_fea_idx] <= fea_val]
        child_X_r = X[X[:, max_fea_idx] > fea_val]
        child_y_r = y[X[:, max_fea_idx] > fea_val]        
        # 构建子树
        child_tree[0] = self._build_tree(child_X_l, child_y_l)
        child_tree[1] = self._build_tree(child_X_r, child_y_r)
        return TreeNode(max_fea_idx, feature_name=feature_name, child=child_tree, feature_val=fea_val)

    def get_entropy(self,y):
        """
        计算熵
        y:
            数据集标签
        """
        entropy = 0
        # 计算每个类别的数量
        num_ck = np.unique(y, return_counts=True)
        for i in range(len(num_ck[0])):
            p = num_ck[1][i] / len(y)
            entropy -= p * np.log2(p)
        return entropy
    def get_conditional_entropy(self, x, y, value):
        """
        计算条件熵
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        cond_entropy = 0
        X_l, X_r, y_l, y_r = self.split(x.reshape(x.shape[0],1),y,0,value=value)
        sub_entropy1 = self.get_entropy(y_l)
        sub_entropy2 = self.get_entropy(y_r)
        p1 = len(y_l) / len(y)
        p2 = len(y_r) / len(y)
        cond_entropy = p1*sub_entropy1 + p2*sub_entropy2
        return cond_entropy
    def get_gain(self, x, y, value):
        """
        计算信息增益
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        return self.get_entropy(y) - self.get_conditional_entropy(x, y,value=value)

    def get_gain_ration(self, x, y,value):
        """
        计算信息增益比
        x:
            数据集的某个特征对应的列向量，训练集中一行表示一个样本，列表示特征
        y:
            数据集标签
        """
        m = self.get_gain(x, y,value=value)
        n = self.get_entropy(x)
        if n != 0:
            return m/n
        else:
            return 0

    def split(self,X,y,d,value):
        index_a = (X[:,d] <= value)
        index_b = (X[:,d] > value)
        return X[index_a], X[index_b], y[index_a], y[index_b]  

    def gini(self,y):
        counter = Counter(y)
        res = 1.0
        for num in counter.values():
            p = num/len(y)
            res -= p**2
        return res

    def try_split(self, X, y, choice=None):
        best_g = -np.inf
        if choice == 'gini':
            best_g = np.inf
        best_d, best_v = -1, -1
        for d in range(X.shape[1]):
            sorted_index = np.argsort(X[:, d])
            for i in range(1,len(X)):
                if X[sorted_index[i-1], d] != X[sorted_index[i], d]:
                    v = (X[sorted_index[i-1], d]+X[sorted_index[i], d])/2 
                    x_l, x_r, y_l, y_r = self.split(X, y, d, v) 
                    length = len(y_l) + len(y_r)
                    if choice == 'gini':
                        g = len(y_l)/length*self.gini(y_l) + len(y_r)/length*self.gini(y_r)
                        if g < best_g:
                            best_g, best_d, best_v = g, d, v
                    elif choice == 'gain':
                        g = self.get_gain(X[:, d], y,value=v)
                        if g > best_g:
                            best_g, best_v, best_d = g, v, d
                    else:
                        g = self.get_gain_ration(X[:, d], y,value=v)
                        if g > best_g:
                            best_g, best_v, best_d = g, v, d                        
        return best_g, best_d, best_v

if __name__ == '__main__':
    data = np.genfromtxt('train.csv',dtype=np.float64,encoding='utf-8-sig',delimiter=',')
    X = data[1:51,:-2]
    y = data[1:51,-1]
    tree = DecisionTreeScratch(etype='ratio',feature_name=np.array(['fixed acidity', ' volatile acidity', ' citric acid', ' residual sugar', ' chlorides', ' free sulfur dioxide', ' total sulfur dioxide', ' density', ' pH', ' sulphates', ' alcohol', ' quality', 'result']))
    root = tree._build_tree(X,y)
    root.DFSearch()
    root.BFSearch()