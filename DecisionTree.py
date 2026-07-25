
import numpy as np
from collections import Counter

class Node:
    def __init__(
            self,
                left=None,
                right=None,
                value=None,
                feature=None,
                threshold=None
    ):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value     
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self,X,y):
        # PS:X.shape[0],means the number of samples(rows);X.shape[1],means the number fo features(columns)
        # deciding to how many features are used to judgement in each node
        # in a certain internal node,the algorithm will randomly pick n_feature features from the avaliable 
        # totally features, then evaluate the possible threshold for each feature and find the one with 
        # highest information gain, then it chooses the overall best split(best feature+its best threshold) from all those candidate
        # this node's decision is only made by this feature and its threshold 
        if not self.n_features:
            self.n_features=X.shape[1]
        else:
            self.n_features=min(X.shape[1],self.n_features)

        self.root=self._grow_tree(X,y)

    def predict(self,X):

        return np.array([self._traverse_tree(sample,self.root)for sample in X])

    def _grow_tree(self,X,y,depth=0):
        n_samples,n_feats=X.shape
        # y is a 1D vector, such as [5,5,4,6,7,...]
        # np.unique(y) tells us how many distinct classes in y
        # for example, if y=[5,5,4,6,7],np.unique(y)=4
        n_labels=len(np.unique(y))
        if(
            # these conditions tell us when to stop growing the tree and return a leaf node
            depth>=self.max_depth
            # this is the pure node condition, which means that there is no benefit in splitting further
            or n_labels==1
            # this is a minimum size constrain to prevent Overfitting
            # in this condition: the current node has fewer samples than min_samples_split, we do not allow a split
            # because if you keep splitting very small groups, the tree will be affected by NOISE in training data
            or n_samples<self.min_samples_split
        ):
            leaf_value=self._most_common_label(y)
            return Node(value=leaf_value)

        # It randomly chooses n_features features from the total feature set
        feat_idxs=np.random.choice(n_feats,self.n_features,replace=False)
        # then finds the best feature and its threshold among them
        best_feat,best_thresh=self._best_split(X,y,feat_idxs)
        # safty check
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        # This creates the two child nodes recursively
        left_idxs,right_idxs=self._split(X[:,best_feat],best_thresh)
        left=self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right=self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left,
            right=right
        )

    def _best_split(self,X,y,feat_idxs):

        best_gain=-1
        split_idx,split_thresh=None,None

        # feat_idxs is the set of randomly selected features in each particular node
        for feat_idx in feat_idxs:
            X_column=X[:,feat_idx] # take one feature column
            thresholds=np.unique(X_column) # all possible split points in this feature
            for threshold in thresholds: # try every possible threshold
                gain=self._information_gain(y,X_column,threshold)
                # keep track of the best one
                if gain>best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_thresh=threshold

        return split_idx,split_thresh

    def _split(self,X_column,threshold):

        left_idxs=np.where(X_column<=threshold)[0]
        right_idxs=np.where(X_column>threshold)[0]
        return left_idxs,right_idxs

    def _information_gain(self,y,X_column,threshold):

        # formula: information gain=H(parent)-[(Nl/N)*Hl+(Nr/N)*Hr]

        # H: entropy; Nl: the number of the samples in left node; N: total samples in current node
        parent_entropy=self._entropy(y)

        # left_idxs: samples where X_column<=threshold
        left_idxs,right_idxs=self._split(X_column,threshold)
        # threshold goes to one side
        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0
        # n: the number of total samples in current node
        n=len(y)
        n_l,n_r=len(left_idxs),len(right_idxs)
        e_l,e_r=self._entropy(y[left_idxs]),self._entropy(y[right_idxs])
        child_entropy=(n_l/n)*e_l+(n_r/n)*e_r 

        return parent_entropy-child_entropy       

    def _entropy(self,y):

        # formula: entropy= -sum(pi*log2(pi))
        # for example, p1 is the proportion of class 1 in all classes in current node
        hist=np.bincount(y)
        ps=hist/len(y)
        return -np.sum([p*np.log2(p) for p in ps if p>0])

    def _most_common_label(self,y):
        # can be converted to numpy version which is needn't to import collections
        counter=Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self,sample,node):
        # making the decision
        if node.is_leaf_node():
            return node.value
        if sample[node.feature]<=node.threshold:
            return self._traverse_tree(sample,node.left)
        return self._traverse_tree(sample,node.right)
    
