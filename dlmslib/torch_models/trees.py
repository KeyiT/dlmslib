

class LabeledTextBinaryTreeNode(object):  # a node in the tree
    def __init__(self, label, word=None):
        self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)

    def __str__(self):
        if self.isLeaf:
            return '[{0}:{1}]'.format(self.word, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.word, self.label, self.right)


class LabeledTextBinaryTree(object):

    def __init__(self, tree_string, open_char='(', close_char=')'):
        tokens = []
        self.open = open_char
        self.close = close_char
        for toks in tree_string.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        self.labels = get_labels(self.root)
        self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        count_open = count_close = 0

        if tokens[split] == self.open:
            count_open += 1
            split += 1
        # Find where left child and right child split
        while count_open != count_close:
            if tokens[split] == self.open:
                count_open += 1
            if tokens[split] == self.close:
                count_close += 1
            split += 1

        # New node
        node = LabeledTextBinaryTreeNode(int(tokens[1]))  # zero index labels

        node.parent = parent

        # leaf Node
        if count_open == 0:
            node.word = ''.join(tokens[2: -1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2: split], parent=node)
        node.right = self.parse(tokens[split: -1], parent=node)

        return node

    def get_words(self):
        leaves = get_leaves(self.root)
        words = [node.word for node in leaves]
        return words


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def get_leaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return get_leaves(node.left) + get_leaves(node.right)


def load_trees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = '../dataset/trees/%s.txt' % dataSet
    print("Loading %s trees.." % dataSet)
    with open(file, 'r', encoding='utf-8') as fid:
        trees = [LabeledTextBinaryTree(l) for l in fid.readlines()]

    return trees
