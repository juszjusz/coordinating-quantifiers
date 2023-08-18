import unittest

from v2.category import NewCategory
from v2.new_guessing_game import NewAgent, NewWord, ConnectionMatrixLxC


class ConnectionMatrixLxC_test(unittest.TestCase):

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
