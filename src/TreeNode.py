class TreeNode:
    def __init__(self, depth=0):
        self.is_leaf = True
        self.weight = 0.0      # w* for leaf
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.depth = depth
        self.gain = 0.0        # gain achieved by split (for importance)