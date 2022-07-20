import numpy as np
import pydot

from smarty.errors import assertion
from smarty.metrics import accuracy
from .utils import print_epoch, print_step, print_info
from .base import BaseSolver, BaseModel

class DecisionNode:
    def __init__(self, gini, col, col_name, condition, samples, left, right):
        self.gini = gini
        self.col = col
        self.col_name = col_name
        self.condition = condition
        self.samples = samples
        self.left = left
        self.right = right

    def copy(self): 
        return DecisionNode(self.gini, self.col, self.col_name, self.condition, self.samples, self.left, self.right)

    def __str__(self):
        return f"'{self.col}' < {self.condition}\n gini: {self.gini}\n samples: {self.samples}"

    def __hash__(self):
        return hash(str(self.gini + self.samples + self.condition))


class TerminalNode:
    def __init__(self, ds, gini, samples):
        targets = ds.get_target_classes()
        distribs = [np.unique(targets[:, idx], return_counts=True) for idx in range(len(ds.target_classes_))]
        self.target = tuple(sorted(zip(u, c), key=lambda x: x[1], reverse=True)[0][0] for u, c in distribs)

        self.gini = gini
        self.samples = samples
        self.ds = ds

    def copy(self):
        return TerminalNode(self.ds, self.gini, self.samples)

    def __str__(self):
        return f"Class '{self.target}'\n gini: {self.gini}\n samples: {self.samples}"

    def __hash__(self):
        return hash(str(self.gini + self.samples) + str(self.target))


class DecisionTreeSolver(BaseSolver):
    def gini_index(self, splits):
        gini = 0.
        size = np.sum([len(split) for split in splits])

        for ds in splits:
            targets = ds.get_target_classes()
            counts = np.array([np.unique(targets[:, idx], return_counts=True)[1] for idx in range(len(ds.target_classes_))])
            proportions = counts / len(ds)
            gini_index = 1 - np.sum(proportions ** 2)
            gini += gini_index * len(ds) / size

        return gini

    def test_split(self, ds, idx, condition):
        left_idxs = np.where(ds.get_data_classes()[:, idx] < condition)[0]
        left_idxs = [int(k) for k in left_idxs]

        if len(left_idxs) != 0:
            left = ds.drop_r(left_idxs)
            return left, ds
        return [ds]

    def get_split(self, ds):
        best_gini = best_condition = best_col = best_splits = None
        data = ds.get_data_classes()

        for col in range(len(ds.data_classes_)):
            class_values = np.unique(data[:, col])

            for condition in class_values:
                splits = self.test_split(ds.copy(), col, condition)
                gini = self.gini_index(splits)

                if best_gini is None or gini < best_gini:
                    best_gini = gini
                    best_condition = condition
                    best_col = col
                    best_splits = splits
        
        return best_gini, best_col, best_condition, best_splits

    def create_node(self, ds, depth, gini, col, condition, left, right=None):
        if right is None or len(left) < self.root.min_samples_ or len(right) < self.root.min_samples_ or depth > self.root.max_depth_:
            return TerminalNode(ds, gini, len(ds))
        return DecisionNode(gini, col, ds.data_classes_[col], condition, len(ds), left, right)

    def create_tree(self, ds, depth):
        gini, col, condition, splits = self.get_split(ds)
        node = self.create_node(ds, depth, gini, col, condition, *splits)

        if not isinstance(node, TerminalNode):
            node.left, left_targets = self.create_tree(node.left, depth + 1)
            node.right, right_targets = self.create_tree(node.right, depth + 1)

            if left_targets == right_targets and len(left_targets) == 1:
                node = TerminalNode(ds, gini, len(ds)) # change node to terminal node chance all its childs predicts same target
                return node, set((node.target, ))
            return node, left_targets.union(right_targets)
        else:
            return node, set((node.target, ))
    
    def fit(self, ds, predict=True, *args, **kwargs):
        print_epoch(1, 1)
        print_step(0, 1)
        self.root.root_node_, _ = self.create_tree(ds, 1)

        kw = {}
        self.fit_predict(predict, ds, kw)
        print_step(1, 1, **kw)

    def predict(self, x_b, *args, **kwargs):
        def helper(node, x):
            if isinstance(node, TerminalNode):
                return node.target
            if x[node.col] < node.condition:
                return helper(node.left, x)
            return helper(node.right, x)

        res = []
        for x in x_b:
            res.append(helper(self.root.root_node_, x))

        return np.array(res)

    def copy_branch(self, root_node, root_node_copy):
        if isinstance(root_node, DecisionNode):
            root_node_copy.left = root_node.left.copy()
            root_node_copy.right = root_node.right.copy()
            self.copy_branch(root_node.left, root_node_copy.left)
            self.copy_branch(root_node.right, root_node_copy.right)

    def get_params(self):
        kw = super().get_params()

        root_node_copy = self.root.root_node_.copy()
        self.copy_branch(self.root.root_node_, root_node_copy)
        return kw.update({
            "root__root_node_": self.root.root_node_,
            "root__max_depth_": self.root.max_depth_,
            "root__min_samples_": self.root.min_samples_,
            "root_loss": self.root.loss
        })


class DecisionTreeClassifier(BaseModel):
    """Decision tree model
    
    :param loss: evaluation loss, has to accept y and y_pred and return score: for pre-defined see smarty.models.metrics
    :param int max_depth: maximum number of splits
    :param int min_samples: minimum number of samples each split must have
    """

    def __init__(self, max_depth=3, min_samples=5, loss=accuracy, *args, **kwargs):
        super(DecisionTreeClassifier, self).__init__(*args, **kwargs)
        self.max_depth_ = max_depth
        self.min_samples_ = min_samples
        self.loss = loss
        self.solver_ = DecisionTreeSolver(self)

    def plot_tree(self, name="my_tree"):
        """Creates tree diagram and saves is as name.png"""
        assertion(self.fitted, "Call .fit() first.")
        graph = pydot.Dot(name, graph_type='digraph', bgcolor='white')

        def helper(root_node, node, label=""):
            if isinstance(node, TerminalNode):
                graph_node = pydot.Node(hash(node), label=str(node), shape="box", style="filled")
            else:
                graph_node = pydot.Node(hash(node), label=str(node), shape="box")
        
            graph.add_node(graph_node)
            graph.add_edge(pydot.Edge(root_node, graph_node, label=label))
            
            if not isinstance(node, TerminalNode):
                helper(graph_node, node.left)
                helper(graph_node, node.right)
        
        graph_node = pydot.Node(hash(self.root_node_), label=str(self.root_node_), shape="box")
        graph.add_node(graph_node)
        if not isinstance(self.root_node_, TerminalNode):
            helper(graph_node, self.root_node_.left, label="Yes")
            helper(graph_node, self.root_node_.right, label="No")
    
        print_info(f"Saving plot to '{name}.png'")
        graph.write_png(f"{name}.png")
        
    def clean_copy(self):
        return DecisionTreeClassifier(self.max_depth_, self.max_samples_)

