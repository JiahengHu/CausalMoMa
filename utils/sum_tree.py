# Adapted from https://github.com/rlcode/per/blob/master/SumTree.py

import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p):
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, dataIdx

    def init_tree(self, ps):
        assert (self.tree == 0).all() and self.write == 0
        assert len(ps) <= self.capacity
        self.tree[self.capacity - 1:self.capacity - 1 + len(ps)] = ps
        self.write = len(ps)

        last_idx = len(ps) - 1 + self.capacity - 1
        last_parent = (last_idx - 1) // 2
        for i in reversed(range(last_parent + 1)):
            left = 2 * i + 1
            right = left + 1
            self.tree[i] = self.tree[left] + self.tree[right]

        assert self.total() == ps.sum()

#
# class ParallelSumTree:
#     def __init__(self, num_trees, capacity):
#         self.num_trees = num_trees
#         self.capacity = capacity
#         self.trees = np.zeros((num_trees, 2 * capacity - 1), dtype=np.float64)
#         self.write = 0
#         self.full = False
#         self.tree_idxes = np.arange(num_trees)
#
#         self.num_updates = 0
#         self.valid_freq = 100000
#
#     # update to the root node
#     def _propagate(self, idxes, changes):
#         # idxes, changes: (num_trees,)
#         zeros = np.zeros_like(changes)
#         while True:
#             idxes = (idxes - 1) // 2
#             self.trees[self.tree_idxes, idxes] += changes
#
#             finish_props = (idxes <= 0)
#             changes = np.where(finish_props, zeros, changes)
#
#             if finish_props.all():
#                 return
#
#     # find sample on leaf node
#     def _retrieve(self, idxes, values):
#         # idxes, value: (num_trees,)
#         tree_len = self.capacity if self.full else self.write
#         tree_len += self.capacity - 1
#         while True:
#             lefts = 2 * idxes + 1
#             rights = lefts + 1
#
#             found_idxes = lefts >= tree_len
#             if found_idxes.all():
#                 return idxes
#
#             modified_lefts = np.where(found_idxes, idxes, lefts)
#             left_values = self.trees[self.tree_idxes, modified_lefts]
#             le_lefts = values <= left_values
#             idxes = np.where(le_lefts, modified_lefts, rights)
#             values = np.where(le_lefts, values, values - left_values)
#
#     def total(self):
#         # return: (num_trees,)
#         return self.trees[:, 0]
#
#     # store priority and sample
#     def add(self, p):
#         raise NotImplementedError
#         idx = self.write + self.capacity - 1
#
#         self.update(idx, p)
#
#         self.write += 1
#         if self.write >= self.capacity:
#             self.full = True
#             self.write = 0
#
#     # update priority
#     def update(self, idxes, ps):
#         # idxes, ps: (num_trees,)
#         changes = ps - self.trees[self.tree_idxes, idxes]
#         self.trees[self.tree_idxes, idxes] = ps
#         self._propagate(idxes, changes)
#
#         self.num_updates += 1
#         if self.num_updates % self.valid_freq == 0:
#             assert np.allclose(self.total, self.trees[:, self.capacity - 1:].sum(axis=-1))
#
#     # get priority and sample
#     def get(self, values):
#         # values: (num_trees,)
#         idxes = self._retrieve(np.zeros(self.num_trees, dtype=np.int32), values)
#         dataIdxes = idxes - self.capacity + 1
#         return idxes, dataIdxes
#
#     def init_trees(self, ps):
#         assert (self.trees == 0).all() and self.write == 0
#         assert len(ps) <= self.capacity
#         self.trees[0, self.capacity - 1:self.capacity - 1 + len(ps)] = ps
#         self.write = len(ps) % self.capacity
#         self.full = len(ps) == self.capacity
#
#         last_idx = len(ps) - 1 + self.capacity - 1
#         last_parent = (last_idx - 1) // 2
#         for i in reversed(range(last_parent + 1)):
#             left = 2 * i + 1
#             right = left + 1
#             self.trees[0, i] = self.trees[0, left] + self.trees[0, right]
#
#         self.trees = np.tile(self.trees[0], (self.num_trees, 1))
#         assert (self.total() == ps.sum()).all()


class ParallelBatchSumTree:
    def __init__(self, num_trees, capacity, batch_size):
        self.num_trees = num_trees
        self.capacity = capacity
        self.batch_size = batch_size

        self.trees = np.zeros((num_trees, 2 * capacity - 1), dtype=np.float64)
        self.write = 0
        self.full = False

        self.tree_idxes = np.tile(np.arange(num_trees)[:, None], (1, batch_size))

    # update to the root node
    def _propagate(self, idxes, changes):
        # idxes, changes: (num_trees, batch_size)
        zeros = np.zeros_like(changes)
        while True:
            idxes = (idxes - 1) // 2

            # similar to self.trees[self.tree_idxes, idxes] += changes but handles repeated inxes
            np.add.at(self.trees, (self.tree_idxes, idxes), changes)

            finish_props = (idxes <= 0)
            changes = np.where(finish_props, zeros, changes)

            if finish_props.all():
                return

    # find sample on leaf node
    def _retrieve(self, idxes, values, monitor):
        # idxes, value: (num_trees, batch_size)
        tree_len = self.capacity if self.full else self.write
        tree_len += self.capacity - 1

        # print(f"tree_len: {tree_len}")
        while True:
            lefts = 2 * idxes + 1
            rights = lefts + 1

            found_idxes = lefts >= tree_len
            if found_idxes.all():
                # if np.any(idxes > tree_len):
                #     import sys
                #     sys.stdout = sys.__stdout__
                #     import ipdb
                #     ipdb.set_trace()
                return idxes

            modified_lefts = np.where(found_idxes, idxes, lefts)
            left_values = self.trees[self.tree_idxes, modified_lefts]
            epsilon = 1e-8
            le_lefts = values <= left_values + epsilon # maybe remove the =? # This might be hacky
            idxes = np.where(le_lefts, modified_lefts, rights)

            if monitor:
                print("Printing retrieve results...")
                print(idxes)
                print(values)
                print(left_values)
                print(le_lefts)
                for i in range(100):
                    print(self.trees[0, 508000 + i * 500: 508010+i*500])
                exit()

            values = np.where(le_lefts, values, values - left_values)

    def total(self):
        return self.trees[:, 0]

    # store priority and sample
    def add(self, p):
        raise NotImplementedError
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.full = True
            self.write = 0

    # update priority
    def update(self, idxes, ps):
        # idxes, ps: (num_trees, batch_size)
        changes = ps - self.trees[self.tree_idxes, idxes]
        self.trees[self.tree_idxes, idxes] = ps
        self._propagate(idxes, changes)

    # get priority and sample
    def get(self, values, monitor=False):
        # values: (num_trees, batch_size)
        idxes = self._retrieve(np.zeros((self.num_trees, self.batch_size), dtype=np.int32), values, monitor)
        dataIdxes = idxes - self.capacity + 1
        return idxes, dataIdxes

    def init_trees(self, ps):
        assert (self.trees == 0).all() and self.write == 0
        assert len(ps) <= self.capacity
        self.trees[:, self.capacity - 1:self.capacity - 1 + len(ps)] = ps
        self.write = len(ps) % self.capacity
        self.full = len(ps) == self.capacity

        last_idx = len(ps) - 1 + self.capacity - 1
        last_parent = (last_idx - 1) // 2
        for i in reversed(range(last_parent + 1)):
            left = 2 * i + 1
            right = left + 1
            self.trees[:, i] = self.trees[:, left] + self.trees[:, right]

        assert (self.total() == ps.sum()).all()