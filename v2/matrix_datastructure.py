import dataclasses
from _bisect import bisect_right
from functools import total_ordering
from typing import Dict, Any, Tuple


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


class One2OneMapping:
    object2index: Dict[Any, int] = {}
    index2object: Dict[int, Tuple[Any, bool]] = {}

    def __len__(self):
        return len(self.object2index)

    def __repr__(self):
        return str(self.object2index)

    def contains(self, element: Any):
        return element in self.object2index.keys()

    def add(self, element: Any):
        index = len(self.object2index)
        self.object2index[element] = index
        self.index2object[index] = (element, True)
        assert len(self.object2index) == len(self.index2object)

    def deactivate_element(self, element: Any):
        index = self.object2index[element]
        self.index2object[index] = (element, False)

    def active_elements(self):
        return [el for el, active in self.index2object.values() if active]

    def nonactive_elements(self):
        return [el for el, active in self.index2object.values() if not active]

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


if __name__ == '__main__':
    mapping = One2OneMapping()
    mapping.add('a')
    mapping.add('aa')
    mapping.add('aaa')
    mapping.add('aaaa')
    mapping.add('aaaaa')
    mapping.add('aaaaaa')
    mapping.add('aaaaaaa')
    mapping.add('aaaaaaaa')
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
