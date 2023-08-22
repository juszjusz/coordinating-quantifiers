import unittest

from v2.category import NewCategory
from v2.new_guessing_game import NewAgent, NewWord, ConnectionMatrixLxC


class ConnectionMatrixLxC_test(unittest.TestCase):

    def setUp(self) -> None:
        self.m = ConnectionMatrixLxC(4)

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
        self.assertEqual(list(r), [0,1])

    def test_ConnectionMatrixLxC(self):
        m = ConnectionMatrixLxC(4)
        m.update_cell(0, 0, lambda v: 1)
        m.update_cell(0, 1, lambda v: 2)
        m.update_cell(0, 2, lambda v: 3)
        m.update_cell(0, 3, lambda v: 4)

        m.update_cell(1, 0, lambda v: 5)
        m.update_cell(1, 1, lambda v: 6)
        m.update_cell(1, 2, lambda v: 7)
        m.update_cell(1, 3, lambda v: 8)

        m.update_cell(2, 0, lambda v: 9)
        m.update_cell(2, 1, lambda v: 10)
        m.update_cell(2, 2, lambda v: 11)
        m.update_cell(2, 3, lambda v: 12)

        m.update_cell(3, 0, lambda v: 13)
        m.update_cell(3, 1, lambda v: 14)
        m.update_cell(3, 2, lambda v: 15)
        m.update_cell(3, 3, lambda v: 16)

        row = 0
        self.assertEqual(m(row, m.get_row_argmax(row)), 4)
        row = 1
        self.assertEqual(m(row, m.get_row_argmax(row)), 8)
        row = 2
        self.assertEqual(m(row, m.get_row_argmax(row)), 12)
        row = 3
        self.assertEqual(m(row, m.get_row_argmax(row)), 16)

        m.update_row(0, [2, 3], -1)
        m.update_column(0, [2, 3], -1)
        print(m.to_ndarray())

    def get_most_connected_word_test(self):
        agent = NewAgent(1, 5)
        w1 = NewWord(1)
        w2 = NewWord(2)
        w3 = NewWord(3)
        w4 = NewWord(4)
        w5 = NewWord(5)

        agent.add_new_word(w1)
        agent.add_new_word(w2)
        agent.add_new_word(w3)
        agent.add_new_word(w4)
        agent.add_new_word(w5)

        # agent.__lxc = None
        category = NewCategory(0, 0)
        agent.add_new_category(category)
        # category_index = NewCategory(1, 20)
        # agent._NewAgent__lxc[1, 2] = 10
        agent.learn_word_category(w1, category, 2)
        agent.learn_word_category(w2, category, 4)
        agent.learn_word_category(w3, category, 3)
        actual = agent.get_most_connected_word(category)
        self.assertEqual(w2, actual)
