from _bisect import bisect_right
from typing import Dict, Any, Tuple


class One2OneMapping:
    object2index: Dict[Any, int] = {}
    index2object: Dict[int, Tuple[Any, bool]] = {}

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

    def remove_nonactive_and_reindex(self):
        active_elements = self.active_elements()
        active_elements = [(el, self.object2index[el]) for el in active_elements]
        active_elements_sorted = list(sorted(active_elements, key=lambda tup: tup[1]))

        for element, index in self.object2index.items():
            _, active = self.index2object[index]
            if not active:

                indices_to_decrement = bisect_right(active_elements_sorted, (element, index), key=lambda tup: tup[1])
                for reindex_element in active_elements_sorted[indices_to_decrement:]:
                    reindex_element[1] -= 1

        self.object2index = {el: index for el, index in active_elements_sorted}
        self.index2object = {index: (el, True) for el, index in self.object2index}
        assert len(self.object2index) == len(self.index2object)


if __name__ == '__main__':
    mapping = One2OneMapping()
    mapping.add('hello')
    mapping.add('xdxd')
    mapping.add('foo')
    mapping.add('coo')
    mapping.deactivate_element('xdxd')
    mapping.remove_nonactive_and_reindex()
    print(mapping)