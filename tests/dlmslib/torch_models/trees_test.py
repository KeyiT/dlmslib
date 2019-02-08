import unittest

import os
from dlmslib.torch_models import trees as trees_module
import numpy as np

import tests


class LabeledTextBinaryTreeNodeTests(unittest.TestCase):

    def setUp(self):
        self.test_ptb_file_path = os.path.join(tests.TEST_ROOT, "resources/torch_models/ptb_trees.txt")

    def test_parse_ptb_string(self):
        trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)
        for tree in trees:
            print(tree)
            self.assertIsNotNone(trees)

    def test_get_transitions(self):
        trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)

        for tree in trees:
            trans = tree.get_transitions()
            print(trans)
            self.assertIsNotNone(trans)

    def test_get_leaf_texts(self):
        trees = trees_module.read_parse_ptb_tree_bank_file(self.test_ptb_file_path)

        for tree in trees:
            texts = tree.get_leaf_texts()
            print(texts)
            self.assertIsNotNone(texts)