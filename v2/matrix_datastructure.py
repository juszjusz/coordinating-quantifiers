import dataclasses
from _bisect import bisect_right
from functools import total_ordering
from typing import Dict, Any, Tuple, List, Callable, Union

import numpy as np
from numpy import ndarray


@dataclasses.dataclass
@total_ordering
class IndexedValue:
    value_index: int
    value: Any

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.value_index == other.value_index and self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Unsupported comparison between different types")
        return self.value_index < other.value_index

    def decrement_index(self):
        self.value_index -= 1

    def __iter__(self):
        return iter((self.value_index, self.value))


@dataclasses.dataclass
class One2OneMapping:
    object2index: Dict[Any, int]
    index2object: Dict[int, Tuple[Any, bool]]

    def __len__(self):
        return len(self.object2index)

    def __repr__(self):
        return str(self.object2index)

    def __copy__(self):
        return One2OneMapping(self.object2index.copy(), self.index2object.copy())

    def get_stored_object(self, obj: Any) -> Any:
        i = self.get_object_index(obj)
        obj, _ = self.get_object_by_index(i)
        return obj

    def get_object_index(self, obj: Any) -> int:
        return self.object2index[obj]

    def get_object_by_index(self, i: int) -> Any:
        return self.index2object[i]

    def contains(self, element: Any):
        return element in self.object2index.keys()

    def add_new_element(self, element: Any):
        index = len(self.object2index)
        self.object2index[element] = index
        self.index2object[index] = (element, True)
        assert len(self.object2index) == len(self.index2object)

    def deactivate_element(self, element: Any):
        index = self.object2index[element]
        self.index2object[index] = (element, False)

    def reactivate_element_at_index(self, i: int):
        el, active = self.get_object_by_index(i)
        assert not active, 'reactivate only previously nonactive object'
        self.index2object[i] = (el, True)

    def active_elements(self) -> List[Any]:
        return [el for el, active in self.index2object.values() if active]

    def nonactive_elements(self) -> List[Any]:
        return [el for el, active in self.index2object.values() if not active]

    def sparsity_rate(self) -> float:
        non_active_size = len(self.nonactive_elements())
        total_size = len(self)
        return non_active_size / total_size if total_size > 0 else 0

    def remove_nonactive_and_reindex(self):
        active_elements = self.active_elements()
        active_elements = [IndexedValue(self.object2index[el], el) for el in active_elements]
        active_elements_sorted = list(sorted(active_elements))
        non_active_elements = self.nonactive_elements()
        non_active_elements = [IndexedValue(self.object2index[el], el) for el in non_active_elements]
        non_active_elements_sorted = list(sorted(non_active_elements, reverse=True))

        for el in non_active_elements_sorted:
            indices_to_decrement = bisect_right(active_elements_sorted, el)
            for reindex_element in active_elements_sorted[indices_to_decrement:]:
                reindex_element.decrement_index()

        self.object2index = {el: index for index, el in active_elements_sorted}
        self.index2object = {index: (el, True) for el, index in self.object2index.items()}
        assert len(self.object2index) == len(self.index2object)
        assert set(self.object2index.values()) == set(range(len(self.object2index)))


class Matrix:
    def __init__(self, max_row: int, max_col: int):
        self._square_matrix = np.zeros((max_row, max_col))
        self._row = 0
        self._col = 0

    def __copy__(self):
        m = Matrix(self._row, self._col)
        m._square_matrix = self.reduce()
        return m

    def __call__(self, row: int, col: int) -> float:
        return self._square_matrix[row, col]

    def rows(self):
        return self._row

    def cols(self):
        return self._col

    def get_rows_all_smaller_than_threshold(self, threshold: float) -> List[int]:
        result, = np.where(np.all(self._square_matrix < threshold, axis=1))
        return [r for r in result if r < self._row]

    def get_row_argmax(self, row_index) -> int:
        return np.argmax(self._square_matrix, axis=1)[row_index]

    def get_col_argmax(self, col_index) -> int:
        return np.argmax(self._square_matrix, axis=0)[col_index]

    def update_cell(self, row: int, column: int, update: Callable[[float], float]):
        recomputed_value = update(self._square_matrix[row, column])
        self._square_matrix[row, column] = recomputed_value

    def update_matrix_on_given_row(self, row_index: int, scalar: float):
        updated_cells = self._square_matrix[row_index, :]
        self._square_matrix[row_index, :] += scalar * updated_cells

    def reset_matrix_on_row_indices(self, row_indices: Union[List[int], ndarray]):
        self._square_matrix[row_indices, :] = 0

    def reset_matrix_on_col_indices(self, col_indices: Union[List[int], ndarray]):
        self._square_matrix[:, col_indices] = 0

    def reduce(self):
        return self._square_matrix[:self._row, :self._col]

    def get_row_vector(self, word_index) -> ndarray:
        row_vector = self._square_matrix[word_index, :]
        adjusted_row_vector = row_vector[:self._col]
        return adjusted_row_vector

    def add_new_row(self):
        self._row += 1
        if self._row == self._square_matrix.shape[0]:
            self._double_height()

    def add_new_col(self):
        self._col += 1
        if self._col == self._square_matrix.shape[1]:
            self._double_width()

    def _double_height(self):
        height, width = self._square_matrix.shape
        new_m = np.zeros((2 * height, width))
        new_m[0:height, :] = self._square_matrix
        self._square_matrix = new_m

    def _double_width(self):
        height, width = self._square_matrix.shape
        new_m = np.zeros((height, width * 2))
        new_m[:, 0:width] = self._square_matrix
        self._square_matrix = new_m

    def remove_columns(self, columns_to_remove: List[int]):
        self._square_matrix = np.delete(self._square_matrix, columns_to_remove, axis=1)
        self._col -= len(columns_to_remove)
        assert self._col >= 0

    def remove_rows(self, rows_to_remove: List[int]):
        self._square_matrix = np.delete(self._square_matrix, rows_to_remove, axis=0)
        self._row -= len(rows_to_remove)
        assert self._row >= 0


if __name__ == '__main__':
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
