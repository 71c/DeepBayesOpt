from abc import abstractmethod
import math
from typing import Iterable, List, Optional, Sequence, Union
import warnings


class _ResizedIterable(Iterable):
    """
    Creates an iterable that resizes given iterable to desired size

    If allow_repeats = False, then:
    Takes any iterable and an integer 'n', and provides an iterator
    that yields the first 'n' elements of the given iterable. If the original
    iterable contains fewer than 'n' elements, the iterator will yield only the
    available  elements without raising an error.
    If allow_repeats = False, then:
    repeats the given iterable until the desired number n.

    Args:
        iterable (iterable): The iterable to wrap.
        n (int): The number of elements to yield from the iterable.
    """
    def __init__(self, iterable, n, allow_repeats=False):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n should be a positive integer.")
        self.iterable = iterable
        self.n = n
        self.allow_repeats = allow_repeats

    def __iter__(self):
        allow_repeats = self.allow_repeats

        n = self.n
        i = 0
        iterator = iter(self.iterable)
        while i < n:
            try:
                yield next(iterator)
                i += 1
            except StopIteration:
                if allow_repeats:
                    iterator = iter(self.iterable)
                else:
                    break

    def __len__(self):
        if self.allow_repeats:
            return self.n
        return min(len_or_inf(self.iterable), self.n)


def resize_iterable(it, new_len: Optional[int] = None, allow_repeats=False):
    if new_len is not None:
        if not isinstance(new_len, int) or new_len <= 0:
            raise ValueError("new_len should be a positive integer")
        original_len = len_or_inf(it)
        if new_len != original_len:
            # Weaker condition than `if isinstance(it, SizedInfiniteIterableMixin):`
            if callable(getattr(it, "copy_with_new_size", None)):
                it = it.copy_with_new_size(new_len)
            else:
                if not allow_repeats and new_len > original_len:
                    raise ValueError(f"{new_len=} should be <= len(it)={original_len} "
                                     "if it is not a SizedInfiniteIterableMixin")
                it = _ResizedIterable(it, new_len, allow_repeats=allow_repeats)
    return it


def iterable_is_finite(x):
    return len_or_inf(x) != math.inf


class SizedIterableMixin(Iterable):
    """An abstract mixin class that provides functionality 'len()' for iterable objects.
    All subclasses should implement `__iter__` because this class inherits from
    the abstract base class `Iterable`.

    Attributes:
        _size (int or inf):
            The size of the iterable object. math.inf if the size is infinite.
            Subclasses must use this attribute to hold the length of the object.
    """
    def _len_or_inf(self):
        if not hasattr(self, "_size"):
            raise AttributeError(
                f"{self.__class__.__name__}, a subclass of SizedIterableMixin, "\
                    "must have attribute '_size' to hold the length.")
        size = self._size
        if size != math.inf and (not isinstance(size, int) or size < 0):
            raise ValueError(
                f"self._size should inf or a non-negative integer but got {size}")
        return size

    def __len__(self):
        size = self._len_or_inf()
        if size == math.inf:
            raise TypeError(f"Length of the {self.__class__.__name__} is infinite")
        return size


def get_lengths_from_proportions(total_length: int, proportions: Sequence[float]):
    # Taken from
    # https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#random_split
    subset_lengths: List[int] = []
    for i, frac in enumerate(proportions):
        if frac < 0 or frac > 1:
            raise ValueError(f"Fraction at index {i} is not between 0 and 1")
        n_items_in_split = int(math.floor(total_length * frac))
        subset_lengths.append(n_items_in_split)
    remainder = total_length - sum(subset_lengths)
    # add 1 to all the lengths in round-robin fashion until the remainder is 0
    for i in range(remainder):
        idx_to_add_at = i % len(subset_lengths)
        subset_lengths[idx_to_add_at] += 1
    lengths = subset_lengths
    for i, length in enumerate(lengths):
        if length == 0:
            warnings.warn(
                f"Length of split at index {i} is 0. "
                f"This might result in an empty dataset."
            )
    return lengths


class SizedInfiniteIterableMixin(SizedIterableMixin):
    """An abstract mixin class that provides functionality for creating iterable objects
    with a specified size. If the size is inf, the object is considered to be
    infinite and so calling iter() then you can call next() indefinitely wihout
    any StopIteration exception.
    If the size is not inf, then the object is considered to be finite and
    calling iter() will return a generator that will yield the next element
    until the size is reached.

    Attributes:
        _size (int or inf):
            The size of the iterable object. math.inf if the size is infinite.
            Subclasses must use this attribute to hold the length of the object.
    """
    @abstractmethod
    def copy_with_new_size(self, size:int) -> "SizedInfiniteIterableMixin":
        """Creates a copy of the object with a new size.
        Should set the _size attribute of the new object to the specified size.

        Args:
            size (int): The new size for the object.

        Returns:
            A new instance of the object with the specified size.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _next(self):
        """Returns the next element in the iterable."""
        pass  # pragma: no cover

    def __iter__(self):
        if self._len_or_inf() == math.inf:
            return self
        # Must separate this in a different function because otherwise,
        # iter will always return a generator, even if self._size == math.inf
        return self._finite_iterator()

    def _finite_iterator(self):
        for _ in range(len(self)):
            yield self._next()

    def __next__(self):
        if self._len_or_inf() == math.inf:
            return self._next()
        raise TypeError(
            f"Cannot call __next__ on a finitely sized {type(self)}. Use iter() first.")

    def random_split(self, lengths: Sequence[Union[int, float]]):
        """Split the dataset into multiple datasets with given lengths.

        Args:
            lengths: List of lengths (integers) or proportions (floats summing to 1)
                    for each split dataset.

        Returns:
            List of new dataset instances with the specified lengths.
        """
        # Same check that pytorch does in torch.utils.data.random_split
        lengths_is_proportions = math.isclose(sum(lengths), 1) and sum(lengths) <= 1

        dataset_size = self._size
        if dataset_size == math.inf:
            if lengths_is_proportions:
                raise ValueError(
                    f"The {self.__class__.__name__} should not be infinite if "
                    "lengths is a list of proportions")
        else:
            if lengths_is_proportions:
                lengths = get_lengths_from_proportions(dataset_size, lengths)

            if sum(lengths) != dataset_size:
                raise ValueError(
                    "Sum of input lengths does not equal the dataset size!")
        return [self.copy_with_new_size(length) for length in lengths]
    

def len_or_inf(x):
    try:
        l = len(x)
        if l != math.inf and (not isinstance(l, int) or l < 0):
            raise ValueError(
                f"len(x) should be inf or a non-negative integer but got {l}")
        return l
    except TypeError:
        # Then it has no length, so we can only say it's infinite if
        # it is an iterable.
        try: # Check if it is an iterable
            iter(x)
            return math.inf
        except TypeError:
            raise TypeError(
                f"Object of type {type(x).__name__} is not iterable so has no length")
