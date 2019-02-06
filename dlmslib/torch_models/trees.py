class LabeledTextBinaryTreeNode(object):  # a node in the tree
    def __init__(self, label, text=None):
        self.label = label
        self.text = text
        self.left = None  # reference to left child
        self.right = None  # reference to right child

    def __str__(self):
        if self.is_leaf():
            return '[{0}:{1}]'.format(self.text, self.label)
        return '({0} <- [{1}:{2}] -> {3})'.format(self.left, self.text, self.label, self.right)

    def is_leaf(self):
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)
        return self.left is None and self.right is None

    def get_leaf_texts(self):
        # from left to right
        if self.is_leaf():
            return [self.text]
        else:
            return self.get_leaf_texts(self.left) + self.get_leaf_texts(self.right)

    @classmethod
    def parse_ptb_string(cls, ptb_string, open_char='(', close_char=')'):
        tokens = []
        for toks in ptb_string.strip().split():
            tokens += list(toks)
        return LabeledTextBinaryTreeNode.__parse_ptb_tokens(tokens, open_char=open_char, close_char=close_char)

    @classmethod
    def __parse_ptb_tokens(cls, tokens, open_char='(', close_char=')'):
        assert tokens[0] == open_char, "Malformed tree"
        assert tokens[-1] == close_char, "Malformed tree"

        split = 2  # position after open and label
        count_open = count_close = 0

        if tokens[split] == open_char:
            count_open += 1
            split += 1
        # Find where left child and right child split
        while count_open != count_close:
            if tokens[split] == open_char:
                count_open += 1
            if tokens[split] == close_char:
                count_close += 1
            split += 1

        # New node
        node = LabeledTextBinaryTreeNode(int(tokens[1]))  # zero index labels

        # leaf Node
        if count_open == 0:
            node.text = ''.join(tokens[2: -1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = LabeledTextBinaryTreeNode.__parse_ptb_tokens(tokens[2: split], open_char=open_char, close_char=close_char)
        node.right = LabeledTextBinaryTreeNode.__parse_ptb_tokens(tokens[split: -1], open_char=open_char, close_char=close_char)

        return node
