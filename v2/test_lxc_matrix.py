import unittest
from typing import List

import numpy as np

from matrix_datastructure import One2OneMapping, Matrix


class One2OneMapping_test(unittest.TestCase):
    def setUp(self) -> None:
        self.r = np.random.RandomState(0)

    def create_random_slices(self, l: int) -> List[int]:
        slice = 0
        slices = [slice]
        while slice != l:
            next_slice = self.r.randint(slice + 1, l + 1)
            slices.append(next_slice)
            slice = next_slice
        return slices

    def test_items_removal(self):
        items = [*range(1000)]
        self.r.shuffle(items)
        mapping = One2OneMapping()
        for i in items:
            mapping.add_new_element(i)

        self.r.shuffle(items)
        indices = self.create_random_slices(len(items))
        slices = [*zip(indices, indices[1:])]
        random_partition = [items[i:j] for i, j in slices]
        for p in random_partition:
            for i in p:
                mapping.deactivate_element(i)
            mapping.remove_nonactive_and_reindex()
            self.assertEqual(len(mapping.nonactive_elements()), 0)
            # self.assertEqual(len(mapping.active_elements()), )
            # indices range densely from 0 to mapping size, i.e. if mapping size = 5, then its indices = 0,1,2,3,4
            self.assertEqual(set(mapping.index2object.keys()), {*range(len(mapping))})

        self.assertEqual(len(mapping), 0)

    def test(self):
        mapping = One2OneMapping()

        mapping.add_new_element('a')
        mapping.add_new_element('aa')
        mapping.add_new_element('aaa')
        mapping.add_new_element('aaaa')
        mapping.add_new_element('aaaaa')
        mapping.add_new_element('aaaaaa')
        mapping.add_new_element('aaaaaaa')
        mapping.add_new_element('aaaaaaaa')
        print(mapping)
        mapping.deactivate_element('a')
        mapping.deactivate_element('aa')
        mapping.deactivate_element('aaaaa')
        mapping.remove_nonactive_and_reindex()
        print(mapping)
        mapping.deactivate_element('aaa')
        mapping.remove_nonactive_and_reindex()
        print(mapping)
        mapping.deactivate_element('aaaaaa')
        mapping.deactivate_element('aaaaaaa')
        mapping.remove_nonactive_and_reindex()
        print(mapping)
        mapping.deactivate_element('aaaa')
        mapping.deactivate_element('aaaaaaaa')
        mapping.remove_nonactive_and_reindex()
        print(mapping)


class ConnectionMatrixLxC_test(unittest.TestCase):

    def setUp(self) -> None:
        self.m = Matrix(4, 4)

        # testing matrix
        # 1   2   3    4
        # 5   100 7    8
        # 200 10  11   12
        # 13  14  1000 16

        self.m.update_cell(0, 0, lambda v: 1)
        self.m.update_cell(0, 1, lambda v: 2)
        self.m.update_cell(0, 2, lambda v: 3)
        self.m.update_cell(0, 3, lambda v: 4)

        self.m.update_cell(1, 0, lambda v: 5)
        self.m.update_cell(1, 1, lambda v: 100)
        self.m.update_cell(1, 2, lambda v: 7)
        self.m.update_cell(1, 3, lambda v: 8)

        self.m.update_cell(2, 0, lambda v: 200)
        self.m.update_cell(2, 1, lambda v: 10)
        self.m.update_cell(2, 2, lambda v: 11)
        self.m.update_cell(2, 3, lambda v: 12)

        self.m.update_cell(3, 0, lambda v: 13)
        self.m.update_cell(3, 1, lambda v: 14)
        self.m.update_cell(3, 2, lambda v: 1000)
        self.m.update_cell(3, 3, lambda v: 16)

    def test_ConnectionMatrixLxC_row_argmax(self):
        # testing matrix
        # 1   2   3    4  row0 argmax 3
        # 5   100 7    8  row1 argmax 1
        # 200 10  11   12 row2 argmax 0
        # 13  14  1000 16 row3 argmax 4

        argmax = self.m.get_row_argmax(row_index=0)
        self.assertEqual(argmax, 3)
        argmax = self.m.get_row_argmax(row_index=1)
        self.assertEqual(argmax, 1)
        argmax = self.m.get_row_argmax(row_index=2)
        self.assertEqual(argmax, 0)
        argmax = self.m.get_row_argmax(row_index=3)
        self.assertEqual(argmax, 2)

    def test_ConnectionMatrixLxC_col_argmax(self):
        # testing matrix
        # 1    2    3    4
        # 5    100  7    8
        # 200  10   11   12
        # 13   14   1000 16
        # col0 col1 col2 col3
        # 2    1    3    3
        argmax = self.m.get_col_argmax(col_index=0)
        self.assertEqual(argmax, 2)
        argmax = self.m.get_col_argmax(col_index=1)
        self.assertEqual(argmax, 1)
        argmax = self.m.get_col_argmax(col_index=2)
        self.assertEqual(argmax, 3)
        argmax = self.m.get_col_argmax(col_index=3)
        self.assertEqual(argmax, 3)

    def test_ConnectionMatrixLxC_get_rows_all_smaller_than_threshold(self):
        # testing matrix
        # 1    2    3    4
        # 5    100  7    8
        # 200  10   11   12
        # 13   14   1000 16
        r = self.m.get_rows_all_smaller_than_threshold(threshold=5)
        self.assertEqual(r, [0])
        r = self.m.get_rows_all_smaller_than_threshold(threshold=101)
        self.assertEqual(list(r), [0, 1])
