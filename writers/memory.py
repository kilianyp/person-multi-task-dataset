"""
# THIS LEAKS MEMORY WTF
class MemoryWriter(object):
    def first_write(self, **data_to_write):
        for key, value in data_to_write.items():
            self.data[key] = [value]

        self.write = self.rest_write

    def rest_write(self, **data_to_write):
        for key, value in data_to_write.items():
            self.data[key].append(value)

    def __enter__(self):
        self.data = {}
        self.write = self.first_write
        return self

    def __exit__(self, *args):
        # keep memory
        pass
"""
import collections
import warnings
class MemoryWriter(object):
    """No leak but still causes problems. Memory is not freed fully."""
    def __init__(self):
        warnings.warn("Do not save huge lists in MemoryWriter. Memory issue.")

    def write(self, **data_to_write):
        for key, value in data_to_write.items():
            self.data[key].append(value)

    def __enter__(self):
        self.data = collections.defaultdict(list)
        return self

    def __exit__(self, *args):
        # keep memory
        pass

