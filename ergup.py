def if__(cond, then, else_=lambda: None):
    if cond:
        return then()
    else:
        return else_()


def for__(iterable, body):
    for i in iterable:
        body(i)


def while__(cond_block, body):
    while cond_block():
        body()


def with__(obj, body):
    with obj as o:
        body(o)

def discard__(obj):
    pass


def assert__(test, msg=None):
    assert test, msg


def then__(x, f):
    if x is None or x is NotImplemented:
        return x
    else:
        return f(x)
# from typing import TypeVar, Union, _SpecialForm, _type_check


class Error:
    def __init__(self, message):
        self.message = message


# T = TypeVar("T")
# @_SpecialForm
# def Result(self, parameters):
#    """Result type.
#
#    Result[T] is equivalent to Union[T, Error].
#    """
#    arg = _type_check(parameters, f"{self} requires a single type.")
#    return [arg, Error]


def is_ok(obj) -> bool:
    return not isinstance(obj, Error)



# from collections.abc import Iterable, Sequence, Iterator, Container


class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, item):
        pass

    @staticmethod
    def from_slice(slice):
        pass

    def into_slice(self):
        pass

    def __getitem__(self, item):
        res = self.start + item
        if res in self:
            return res
        else:
            raise IndexError("Index out of range")

    # TODO: for Str, etc.
    def __len__(self):
        if self.start in self:
            if self.end in self:
                # len(1..4) == 4
                return self.end - self.start + 1
            else:
                # len(1..<4) == 3
                return self.end - self.start
        else:
            if self.end in self:
                # len(1<..4) == 3
                return self.end - self.start
            else:
                # len(1<..<4) == 2
                return self.end - self.start - 2

    def __iter__(self):
        return RangeIterator(rng=self)


# Sequence.register(Range)
# Container.register(Range)
# Iterable.register(Range)


# represents `start<..end`
class LeftOpenRange(Range):
    def __contains__(self, item):
        return self.start < item <= self.end


# represents `start..<end`
class RightOpenRange(Range):
    def __contains__(self, item):
        return self.start <= item < self.end

    @staticmethod
    def from_slice(slice):
        return Range(slice.start, slice.stop)

    def into_slice(self):
        return slice(self.start, self.end)


# represents `start<..<end`
class OpenRange(Range):
    def __contains__(self, item):
        return self.start < item < self.end


# represents `start..end`
class ClosedRange(Range):
    def __contains__(self, item):
        return self.start <= item <= self.end

    @staticmethod
    def from_slice(slice):
        return Range(slice.start, slice.stop - 1)

    def into_slice(self):
        return slice(self.start, self.end + 1)


class RangeIterator:
    def __init__(self, rng):
        self.rng = rng
        self.needle = self.rng.start
        if issubclass(Nat, type(self.rng.start)):
            if self.needle not in self.rng:
                self.needle += 1
        elif issubclass(Str, type(self.rng.start)):
            if self.needle not in self.rng:
                self.needle = chr(ord(self.needle) + 1)
        else:
            if self.needle not in self.rng:
                self.needle = self.needle.succ()

    def __iter__(self):
        return self

    def __next__(self):
        if issubclass(Nat, type(self.rng.start)):
            if self.needle in self.rng:
                result = self.needle
                self.needle += 1
                return result
        elif issubclass(Str, type(self.rng.start)):
            if self.needle in self.rng:
                result = self.needle
                self.needle = chr(ord(self.needle) + 1)
                return result
        else:
            if self.needle in self.rng:
                result = self.needle
                self.needle = self.needle.succ()
                return result
        raise StopIteration


# Iterator.register(RangeIterator)
try:
    from typing import Union
except ImportError:
    import warnings

    warnings.warn("`typing.Union` is not available. Please use Python 3.8+.")

    class Union:
        pass


class UnionType:
    __origin__ = Union
    __args__: list  # list[type]

    def __init__(self, *args):
        self.__args__ = args

    def __str__(self):
        s = "UnionType[" + ", ".join(str(arg) for arg in self.__args__) + "]"
        return s

    def __repr__(self):
        return self.__str__()


class FakeGenericAlias:
    __origin__: type
    __args__: list  # list[type]

    def __init__(self, origin, *args):
        self.__origin__ = origin
        self.__args__ = args


try:
    from types import GenericAlias
except ImportError:
    GenericAlias = FakeGenericAlias


def is_type(x) -> bool:
    return isinstance(x, (type, FakeGenericAlias, GenericAlias, UnionType))


# The behavior of `builtins.isinstance` depends on the Python version.
def _isinstance(obj, classinfo) -> bool:
    if isinstance(classinfo, (FakeGenericAlias, GenericAlias, UnionType)):
        if classinfo.__origin__ == Union:
            return any(_isinstance(obj, t) for t in classinfo.__args__)
        else:
            return isinstance(obj, classinfo.__origin__)
    else:
        try:
            return isinstance(obj, classinfo)
        except:
            return False


class MutType:
    value: object

    # This method is a fallback to implement pseudo-inheritance.
    def __getattr__(self, name):
        return object.__getattribute__(self.value, name)
from collections import namedtuple







# (elem in y) == contains_operator(y, elem)
def contains_operator(y, elem) -> bool:
    if hasattr(elem, "type_check"):
        return elem.type_check(y)
    elif isinstance(y, UnionType):
        return any([contains_operator(t, elem) for t in y.__args__])
    # 1 in Int
    elif is_type(y):
        if _isinstance(elem, y):
            return True
        elif hasattr(y, "generic_try_new"):
            return is_ok(y.generic_try_new(elem, y))
        elif hasattr(y, "try_new") and is_ok(y.try_new(elem)):
            return True
        elif hasattr(y, "__origin__") and hasattr(y.__origin__, "type_check"):
            return y.__origin__.type_check(elem, y)
        # TODO: trait check
        return False
    # [1] in [Int]
    elif (
        _isinstance(y, list)
        and _isinstance(elem, list)
        and (len(y) == 0 or is_type(y[0]) or _isinstance(y[0], Range))
    ):
        type_check = all(map(lambda x: contains_operator(x[0], x[1]), zip(y, elem)))
        len_check = len(elem) <= len(y)
        return type_check and len_check
    # (1, 2) in (Int, Int)
    elif (
        _isinstance(y, tuple)
        and _isinstance(elem, tuple)
        and (len(y) == 0 or is_type(y[0]) or _isinstance(y[0], Range))
    ):
        if not hasattr(elem, "__iter__"):
            return False
        type_check = all(map(lambda x: contains_operator(x[0], x[1]), zip(y, elem)))
        len_check = len(elem) <= len(y)
        return type_check and len_check
    # {1: 2} in {Int: Int}
    elif (
        _isinstance(y, dict)
        and _isinstance(elem, dict)
        and (len(y) == 0 or is_type(next(iter(y.keys()))))
    ):
        if len(y) == 1:
            key = next(iter(y.keys()))
            key_check = all([contains_operator(key, el) for el in elem.keys()])
            value = next(iter(y.values()))
            value_check = all([contains_operator(value, el) for el in elem.values()])
            return key_check and value_check
        type_check = True  # TODO:
        len_check = True  # It can be True even if either elem or y has the larger number of elems
        return type_check and len_check
    elif _isinstance(elem, list):
        

        return contains_operator(y, List(elem))
    elif callable(elem):
        # TODO:
        return callable(y)
    else:
        return elem in y





class Int(int):
    def try_new(i):  # -> Result[Nat]
        if isinstance(i, int):
            return Int(i)
        else:
            return Error("not an integer")

    def bit_count(self):
        if hasattr(int, "bit_count"):
            return int.bit_count(self)
        else:
            return bin(self).count("1")

    def succ(self):
        return Int(self + 1)

    def pred(self):
        return Int(self - 1)

    def mutate(self):
        return IntMut(self)

    def __add__(self, other):
        return then__(int.__add__(self, other), Int)

    def __sub__(self, other):
        return then__(int.__sub__(self, other), Int)

    def __mul__(self, other):
        return then__(int.__mul__(self, other), Int)

    def __div__(self, other):
        return then__(int.__div__(self, other), Int)

    def __floordiv__(self, other):
        return then__(int.__floordiv__(self, other), Int)

    def __pow__(self, other):
        return then__(int.__pow__(self, other), Int)

    def __rpow__(self, other):
        return then__(int.__pow__(other, self), Int)

    def __pos__(self):
        return self

    def __neg__(self):
        return then__(int.__neg__(self), Int)


class IntMut(MutType):  # inherits Int
    value: Int

    def __init__(self, i):
        self.value = Int(i)

    def __int__(self):
        return self.value.__int__()

    def __float__(self):
        return self.value.__float__()

    def __repr__(self):
        return self.value.__repr__()

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, MutType):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, MutType):
            return self.value != other.value
        else:
            return self.value != other

    def __le__(self, other):
        if isinstance(other, MutType):
            return self.value <= other.value
        else:
            return self.value <= other

    def __ge__(self, other):
        if isinstance(other, MutType):
            return self.value >= other.value
        else:
            return self.value >= other

    def __lt__(self, other):
        if isinstance(other, MutType):
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        if isinstance(other, MutType):
            return self.value > other.value
        else:
            return self.value > other

    def __add__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value + other.value)
        else:
            return IntMut(self.value + other)

    def __sub__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value - other.value)
        else:
            return IntMut(self.value - other)

    def __mul__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value * other.value)
        else:
            return IntMut(self.value * other)

    def __floordiv__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value // other.value)
        else:
            return IntMut(self.value // other)

    def __truediv__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value / other.value)
        else:
            return IntMut(self.value / other)

    def __pow__(self, other):
        if isinstance(other, MutType):
            return IntMut(self.value**other.value)
        else:
            return IntMut(self.value**other)

    def __pos__(self):
        return self

    def __neg__(self):
        return IntMut(-self.value)

    def update(self, f):
        self.value = Int(f(self.value))

    def inc(self, i=1):
        self.value = Int(self.value + i)

    def dec(self, i=1):
        self.value = Int(self.value - i)

    def succ(self):
        return self.value.succ()

    def pred(self):
        return self.value.pred()

    def copy(self):
        return IntMut(self.value)

  # don't unify with the above line





class Nat(Int):
    def __init__(self, i):
        if int(i) < 0:
            raise ValueError("Nat can't be negative: {}".format(i))

    def try_new(i):  # -> Result[Nat]
        if i >= 0:
            return Nat(i)
        else:
            return Error("Nat can't be negative: {}".format(i))

    def times(self, f):
        for _ in range(self):
            f()

    def saturating_sub(self, other):
        if self > other:
            return self - other
        else:
            return 0

    def mutate(self):
        return NatMut(self)

    def __add__(self, other):
        return then__(super().__add__(other), Nat)

    def __mul__(self, other):
        return then__(super().__mul__(other), Nat)

    def __pos__(self):
        return self


class NatMut(IntMut):  # and Nat
    value: Nat

    def __init__(self, n: Nat):
        if int(n) < 0:
            raise ValueError("Nat can't be negative: {}".format(n))
        self.value = n

    def __int__(self):
        return self.value.__int__()

    def __float__(self):
        return self.value.__float__()

    def __repr__(self):
        return self.value.__repr__()

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, MutType):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, MutType):
            return self.value != other.value
        else:
            return self.value != other

    def __le__(self, other):
        if isinstance(other, MutType):
            return self.value <= other.value
        else:
            return self.value <= other

    def __ge__(self, other):
        if isinstance(other, MutType):
            return self.value >= other.value
        else:
            return self.value >= other

    def __lt__(self, other):
        if isinstance(other, MutType):
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        if isinstance(other, MutType):
            return self.value > other.value
        else:
            return self.value > other

    def __add__(self, other):
        if isinstance(other, MutType):
            return NatMut(self.value + other.value)
        else:
            return NatMut(self.value + other)

    def __radd__(self, other):
        if isinstance(other, MutType):
            return Nat(other.value + self.value)
        else:
            return Nat(other + self.value)

    def __mul__(self, other):
        if isinstance(other, MutType):
            return NatMut(self.value * other.value)
        else:
            return NatMut(self.value * other)

    def __rmul__(self, other):
        if isinstance(other, MutType):
            return Nat(other.value * self.value)
        else:
            return Nat(other * self.value)

    def __truediv__(self, other):
        if isinstance(other, MutType):
            return NatMut(self.value / other.value)
        else:
            return NatMut(self.value / other)

    def __pow__(self, other):
        if isinstance(other, MutType):
            return NatMut(self.value**other.value)
        else:
            return NatMut(self.value**other)

    def __pos__(self):
        return self

    def update(self, f):
        self.value = Nat(f(self.value))

    def try_new(i):  # -> Result[Nat]
        if i >= 0:
            return NatMut(i)
        else:
            return Error("Nat can't be negative")

    def times(self, f):
        for _ in range(self.value):
            f()

    def copy(self):
        return NatMut(self.value)






class Bool(Nat):
    def try_new(b: bool):  # -> Result[Nat]
        if isinstance(b, bool):
            return Bool(b)
        else:
            return Error("Bool can't be other than True or False")

    def __str__(self) -> str:
        if self:
            return "True"
        else:
            return "False"

    def __repr__(self) -> str:
        return self.__str__()

    def mutate(self):
        return BoolMut(self)

    def invert(self):
        return Bool(not self)


class BoolMut(NatMut):
    value: Bool

    def __init__(self, b: Bool):
        self.value = b

    def __repr__(self):
        return self.value.__repr__()

    def __bool__(self):
        return bool(self.value)

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, MutType):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, MutType):
            return self.value != other.value
        else:
            return self.value != other

    def update(self, f):
        self.value = Bool(f(self.value))

    def invert(self):
        self.value = self.value.invert()

    def copy(self):
        return BoolMut(self.value)






class Str(str):
    def __instancecheck__(cls, obj):
        return isinstance(obj, str)

    def try_new(s: str):  # -> Result[Nat]
        if isinstance(s, str):
            return Str(s)
        else:
            return Error("Str can't be other than str")

    def get(self, i: int):
        if len(self) > i:
            return Str(self[i])
        else:
            return None

    def mutate(self):
        return StrMut(self)

    def to_int(self):
        return Int(self) if self.isdigit() else None

    def contains(self, s):
        return s in self

    def __add__(self, other):
        return then__(str.__add__(self, other), Str)

    def __mul__(self, other):
        return then__(str.__mul__(self, other), Str)

    def __mod__(self, other):
        return then__(str.__mod__(other, self), Str)

    def __getitem__(self, index_or_slice):
        

        if isinstance(index_or_slice, slice):
            return Str(str.__getitem__(self, index_or_slice))
        elif isinstance(index_or_slice, Range):
            return Str(str.__getitem__(self, index_or_slice.into_slice()))
        else:
            return str.__getitem__(self, index_or_slice)

    def from_(self, nth: int):
        return self[nth:]

class StrMut(MutType):  # Inherits Str
    value: Str

    def __init__(self, s: str):
        self.value = s

    def __repr__(self):
        return self.value.__repr__()

    def __str__(self):
        return self.value.__str__()

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, MutType):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, MutType):
            return self.value != other.value
        else:
            return self.value != other

    def update(self, f):
        self.value = Str(f(self.value))

    def try_new(s: str):
        if isinstance(s, str):
            self = StrMut()
            self.value = s
            return self
        else:
            return Error("Str! can't be other than str")

    def clear(self):
        self.value = ""

    def pop(self):
        if len(self.value) > 0:
            last = self.value[-1]
            self.value = self.value[:-1]
            return last
        else:
            return Error("Can't pop from empty `Str!`")

    def push(self, s: str):
        self.value += s

    def remove(self, idx: int):
        char = self.value[idx]
        self.value = self.value[:idx] + self.value[idx + 1 :]
        return char

    def insert(self, idx: int, s: str):
        self.value = self.value[:idx] + s + self.value[idx:]

    def copy(self):
        return StrMut(self.value)





class Float(float):
    EPSILON = 2.220446049250313e-16

    def try_new(i):  # -> Result[Nat]
        if isinstance(i, float):
            return Float(i)
        else:
            return Error("not a float")

    def mutate(self):
        return FloatMut(self)

    def __abs__(self):
        return Float(float.__abs__(self))

    def __add__(self, other):
        return then__(float.__add__(self, other), Float)

    def __sub__(self, other):
        return then__(float.__sub__(self, other), Float)

    def __mul__(self, other):
        return then__(float.__mul__(self, other), Float)

    def __div__(self, other):
        return then__(float.__div__(self, other), Float)

    def __floordiv__(self, other):
        return then__(float.__floordiv__(self, other), Float)

    def __truediv__(self, other):
        return then__(float.__truediv__(self, other), Float)

    def __pow__(self, other):
        return then__(float.__pow__(self, other), Float)

    def __rpow__(self, other):
        return then__(float.__pow__(float(other), self), Float)

    def __pos__(self):
        return self

    def __neg__(self):
        return then__(float.__neg__(self), Float)

    def nearly_eq(self, other, epsilon=EPSILON):
        return abs(self - other) < epsilon


class FloatMut(MutType):  # inherits Float
    value: Float

    EPSILON = 2.220446049250313e-16

    def __init__(self, i):
        self.value = Float(i)

    def __repr__(self):
        return self.value.__repr__()

    def __hash__(self):
        return self.value.__hash__()

    def __deref__(self):
        return self.value

    def __float__(self):
        return self.value.__float__()

    def __eq__(self, other):
        if isinstance(other, MutType):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other):
        if isinstance(other, MutType):
            return self.value != other.value
        else:
            return self.value != other

    def __le__(self, other):
        if isinstance(other, MutType):
            return self.value <= other.value
        else:
            return self.value <= other

    def __ge__(self, other):
        if isinstance(other, MutType):
            return self.value >= other.value
        else:
            return self.value >= other

    def __lt__(self, other):
        if isinstance(other, MutType):
            return self.value < other.value
        else:
            return self.value < other

    def __gt__(self, other):
        if isinstance(other, MutType):
            return self.value > other.value
        else:
            return self.value > other

    def __add__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value + other.value)
        else:
            return FloatMut(self.value + other)

    def __sub__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value - other.value)
        else:
            return FloatMut(self.value - other)

    def __mul__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value * other.value)
        else:
            return FloatMut(self.value * other)

    def __floordiv__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value // other.value)
        else:
            return FloatMut(self.value // other)

    def __truediv__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value / other.value)
        else:
            return FloatMut(self.value / other)

    def __pow__(self, other):
        if isinstance(other, MutType):
            return FloatMut(self.value**other.value)
        else:
            return FloatMut(self.value**other)

    def __pos__(self):
        return self

    def __neg__(self):
        return FloatMut(-self.value)

    def update(self, f):
        self.value = Float(f(self.value))

    def inc(self, value=1.0):
        self.value = Float(self.value + value)

    def dec(self, value=1.0):
        self.value = Float(self.value - value)

    def copy(self):
        return FloatMut(self.value)










class List(list):
    @staticmethod
    def try_new(lis):  # -> Result[List]
        if isinstance(lis, list):
            return List(lis)
        else:
            return Error("not a list")

    def generic_try_new(lis, cls=None):  # -> Result[List]
        if cls is None:
            return List.try_new(lis)
        else:
            elem_t = cls.__args__[0]
            elems = []
            for elem in lis:
                if not hasattr(elem_t, "try_new"):
                    return Error("not a " + str(elem_t))
                # TODO: nested check
                elem = elem_t.try_new(elem)
                if is_ok(elem):
                    elems.append(elem)
                else:
                    return Error("not a " + str(elem_t))
            return List(elems)

    def dedup(self, same_bucket=None):
        if same_bucket is None:
            return List(list(set(self)))
        else:
            removes = []
            for lhs, rhs in zip(self, self[1:]):
                if same_bucket(lhs, rhs):
                    removes.append(lhs)
            for remove in removes:
                self.remove(remove)
            return self

    def get(self, index, default=None):
        try:
            return self[index]
        except IndexError:
            return default

    def push(self, value):
        self.append(value)
        return self

    def partition(self, f):
        return List(list(filter(f, self))), List(
            list(filter(lambda x: not f(x), self))
        )

    def __mul__(self, n):
        return then__(list.__mul__(self, n), List)

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, slice):
            return List(list.__getitem__(self, index_or_slice))
        elif isinstance(index_or_slice, NatMut) or isinstance(index_or_slice, IntMut):
            return list.__getitem__(self, int(index_or_slice))
        elif isinstance(index_or_slice, Range):
            return List(list.__getitem__(self, index_or_slice.into_slice()))
        else:
            return list.__getitem__(self, index_or_slice)

    def __hash__(self):
        return hash(tuple(self))

    def update(self, f):
        self = List(f(self))

    def type_check(self, t: type) -> bool:
        if isinstance(t, list):
            if len(t) < len(self):
                return False
            for inner_t, elem in zip(t, self):
                if not contains_operator(inner_t, elem):
                    return False
            return True
        elif isinstance(t, set):
            return self in t
        elif isinstance(t, UnionType):
            return any([self.type_check(_t) for _t in t.__args__])
        elif not hasattr(t, "__args__"):
            return isinstance(self, t)
        elem_t = t.__args__[0]
        l = None if len(t.__args__) != 2 else t.__args__[1]
        if l is not None and l != len(self):
            return False
        for elem in self:
            if not contains_operator(elem_t, elem):
                return False
        return True

    def update_nth(self, index, f):
        self[index] = f(self[index])

    def sum(self, start=0):
        return sum(self, start)

    def prod(self, start=1):
        from functools import reduce

        return reduce(lambda x, y: x * y, self, start)

    def reversed(self):
        return List(list.__reversed__(self))

    def insert_at(self, index, value):
        self.insert(index, value)
        return self

    def remove_at(self, index):
        del self[index]
        return self

    def remove_all(self, item):
        while item in self:
            self.remove(item)
        return self

    def repeat(self, n):
        from copy import deepcopy

        new = []
        for _ in range(n):
            new.extend(deepcopy(self))
        return List(new)

    def from_(self, nth: int):
        return self[nth:]

class UnsizedList:
    elem: object

    def __init__(self, elem):
        self.elem = elem


class Dict(dict):
    @staticmethod
    def try_new(dic):  # -> Result[Dict]
        if isinstance(dic, dict):
            return Dict(dic)
        else:
            return Error("not a dict")

    def concat(self, other):
        return Dict({**self, **other})

    def diff(self, other):
        return Dict({k: v for k, v in self.items() if k not in other})

    # other: Iterable
    def update(self, other, conflict_resolver=None):
        if conflict_resolver is None:
            super().update(other)
        elif isinstance(other, dict):
            self.merge(other, conflict_resolver)
        else:
            for k, v in other:
                if k in self:
                    self[k] = conflict_resolver(self[k], v)
                else:
                    self[k] = v

    # other: Dict
    def merge(self, other, conflict_resolver=None):
        self.update(other, conflict_resolver)

    def insert(self, key, value):
        self[key] = value

    def remove(self, key):
        res = self.get(key)
        if res is not None:
            del self[key]
        return res

    def as_record(self):
        from collections import namedtuple

        return namedtuple("Record", self.keys())(**self)
class Set(set):
    pass




class Bytes(bytes):
    def try_new(b):  # -> Result[Nat]
        if isinstance(b, bytes):
            return Bytes(bytes(b))
        else:
            return Error("not a bytes")

    def __getitem__(self, index_or_slice):
        if isinstance(index_or_slice, slice):
            return Bytes(bytes.__getitem__(self, index_or_slice))
        elif isinstance(index_or_slice, Range):
            return Bytes(bytes.__getitem__(self, index_or_slice.into_slice()))
        else:
            return bytes.__getitem__(self, index_or_slice)
os_L6 = __import__("os.path")
os_L6 = __import__("os.path")






def int__(i):
    return Int(i)


def nat__(i):
    return Nat(i)


def float__(f):
    return Float(f)


def str__(s):
    return Str(s)
def if_tmp_func_1__():
    if (Int((res2_L27_C4).returncode) != Nat(0)):
        assert contains_operator(Bytes,(res2_L27_C4).stderr)
        if_tmp_0__ = (quit)(((Str("Failed to install poise: ") + (str__)((Bytes((res2_L27_C4).stderr)).decode(),)) + Str("")),)
    else:
        if_tmp_0__ = None
    return if_tmp_0__
def mutate_operator(x):
    if hasattr(x, "mutate"):
        return x.mutate()
    else:
        return x
os_L6 = __import__("os.path")
def if_tmp_func_10__():
    if (Str(answer_L47_C16) == Str("y")):
        (install_poise__erg_proc___L16)()
        if_tmp_9__ = (exit)(Nat(0),)
    else:
        if_tmp_9__ = None
    return if_tmp_9__
def if_tmp_func_8__():
    if (Int(((sub_L9).run(Str("poise"),capture_output=Bool(True),shell=Bool(True),)).returncode) != Nat(0)):
        (print)(Str("poise is not installed, do you want to install it? [y/n]"),end=Str(" "),)
        global answer_L47_C16
        answer_L47_C16 = (input)()
        if_tmp_7__ = if_tmp_func_10__()
    else:
        if_tmp_7__ = None
    return if_tmp_7__
def if_tmp_func_6__():
    if overwrite_L35:
        (print)(((Str("Removing ") + (str__)(Str(erg_dir_L12),)) + Str(" ...")),)
        if_tmp_5__ = (su_L8).rmtree(Str(erg_dir_L12),)
    else:
        if_tmp_func_8__()
        (print)(Str("Aborting installation"),)
        if_tmp_5__ = (exit)(Nat(1),)
    return if_tmp_5__
def if_tmp_func_3__():
    if (
(os_L6).path
).exists(Str(erg_dir_L12),):
        (print)(Str(".erg directory already exists, do you want to overwrite it? [y/n]"),end=Str(" "),)
        global answer_L38_C4
        answer_L38_C4 = (input)()
        (overwrite_L35).update((lambda _4,:            (Str(answer_L38_C4) == Str("y"))),)
        if_tmp_2__ = if_tmp_func_6__()
    else:
        if_tmp_2__ = None
    return if_tmp_2__
urllib_L1 = __import__("urllib.request")
urllib_L1 = __import__("urllib.request")
def if_tmp_func_12__():
    if ((List((sys_L5).argv)).get(Nat(1),) == Str("nightly")):
        global latest_url_L59_C8
        latest_url_L59_C8 = Str("https://api.github.com/repos/erg-lang/erg/releases")
        global _stream_L60_C8
        _stream_L60_C8 = (
        (urllib_L1).request
        ).urlopen(Str(latest_url_L59_C8),)
        global s_L61_C8
        s_L61_C8 = ((_stream_L60_C8).read()).decode()
        global jdata_L62_C8
        jdata_L62_C8 = (json_L4).loads(Str(s_L61_C8),)
        assert contains_operator((List)[Dict({(Str): (object),}),],jdata_L62_C8)
        if_tmp_11__ = ((List(jdata_L62_C8)).__getitem__(Nat(0),)).__getitem__(Str("tag_name"),)
    else:
        global latest_url_L66_C8
        latest_url_L66_C8 = Str("https://api.github.com/repos/erg-lang/erg/releases/latest")
        global _stream_L67_C8
        _stream_L67_C8 = (
        (urllib_L1).request
        ).urlopen(Str(latest_url_L66_C8),)
        global s_L68_C8
        s_L68_C8 = ((_stream_L67_C8).read()).decode()
        global jdata_L69_C8
        jdata_L69_C8 = (json_L4).loads(Str(s_L68_C8),)
        assert contains_operator(Dict({(Str): (object),}),jdata_L69_C8)
        if_tmp_11__ = (Dict(jdata_L69_C8)).__getitem__(Str("tag_name"),)
    return if_tmp_11__
def match_tmp_func_14__():
    match Str((sys_L5).platform):
        case ("darwin") as __percent__p_desugar_1_L76_C4:
            match_tmp_13__ = Str("erg-x86_64-apple-darwin.tar.gz")
        case ("win32") as __percent__p_desugar_2_L77_C4:
            match_tmp_13__ = Str("erg-x86_64-pc-windows-msvc.zip")
        case _:
            match_tmp_13__ = Str("erg-x86_64-unknown-linux-gnu.tar.gz")
    return match_tmp_13__
urllib_L1 = __import__("urllib.request")
def if_tmp_func_16__():
    if (Str((sys_L5).platform) == Str("win32")):
        (print)(((Str("Extracting ") + (str__)(Str(filename_L75),)) + Str(" ...")),)
        global bytesio_L87_C8
        bytesio_L87_C8 = (io_L7).BytesIO((stream_L83).read(),)
        global zipfile_L88_C8
        zipfile_L88_C8 = (zf_L3).ZipFile(bytesio_L87_C8,)
        (zipfile_L88_C8).extractall(Str(erg_tmp_dir_L14),)
        (zipfile_L88_C8).close()
        (discard__)((su_L8).move(((Str("") + (str__)(Str(erg_tmp_dir_L14),)) + Str("/erg.exe")),((Str("") + (str__)(Str(erg_bin_dir_L13),)) + Str("/erg.exe")),),)
        (discard__)((su_L8).move(((Str("") + (str__)(Str(erg_tmp_dir_L14),)) + Str("/lib")),((Str("") + (str__)(Str(erg_dir_L12),)) + Str("/lib")),),)
        if_tmp_15__ = (su_L8).rmtree(Str(erg_tmp_dir_L14),)
    else:
        (print)(((Str("Extracting ") + (str__)(Str(filename_L75),)) + Str(" ...")),)
        global tarfile_L96_C8
        tarfile_L96_C8 = (tf_L2).open(fileobj=stream_L83,mode=Str("r|gz"),)
        (tarfile_L96_C8).extractall(Str(erg_tmp_dir_L14),)
        (tarfile_L96_C8).close()
        (discard__)((su_L8).move(((Str("") + (str__)(Str(erg_tmp_dir_L14),)) + Str("/erg")),((Str("") + (str__)(Str(erg_bin_dir_L13),)) + Str("/erg")),),)
        (discard__)((su_L8).move(((Str("") + (str__)(Str(erg_tmp_dir_L14),)) + Str("/lib")),((Str("") + (str__)(Str(erg_dir_L12),)) + Str("/lib")),),)
        if_tmp_15__ = (su_L8).rmtree(Str(erg_tmp_dir_L14),)
    return if_tmp_15__
urllib_L1 = (__import__)(Str("urllib"),)
tf_L2 = (__import__)(Str("tarfile"),)
zf_L3 = (__import__)(Str("zipfile"),)
json_L4 = (__import__)(Str("json"),)
sys_L5 = (__import__)(Str("sys"),)
os_L6 = (__import__)(Str("os"),)
io_L7 = (__import__)(Str("io"),)
su_L8 = (__import__)(Str("shutil"),)
sub_L9 = (__import__)(Str("subprocess"),)
homedir_L11 = (
(os_L6).path
).expanduser(Str("~"),)
erg_dir_L12 = (Str(homedir_L11) + Str("/.erg"))
erg_bin_dir_L13 = (Str(homedir_L11) + Str("/.erg/bin"))
erg_tmp_dir_L14 = (Str(homedir_L11) + Str("/.erg/tmp"))
def install_poise__erg_proc___L16():
    global poise_git_url_L17_C4
    poise_git_url_L17_C4 = Str("https://github.com/erg-lang/poise.git")
    (print)(Str("Cloning poise (erg package manager) ..."),)
    (os_L6).mkdir(Str(erg_tmp_dir_L14),) if (not ((
    (os_L6).path
    ).exists(Str(erg_tmp_dir_L14),))) else None
    (os_L6).chdir(Str(erg_tmp_dir_L14),)
    global res_L22_C4
    res_L22_C4 = (sub_L9).run(List([Str("git"),Str("clone"),Str(poise_git_url_L17_C4),]),capture_output=Bool(True),)
    (quit)(Str("Failed to clone poise repo"),) if (Int((res_L22_C4).returncode) != Nat(0)) else None
    (os_L6).chdir(Str("poise"),)
    (print)(Str("Building poise ..."),)
    global res2_L27_C4
    res2_L27_C4 = (sub_L9).run(List([((Str("") + (str__)(Str(erg_bin_dir_L13),)) + Str("/erg")),Str("src/main.er"),Str("--"),Str("install"),]),capture_output=Bool(True),)
    if_tmp_func_1__()
    (print)(Str("poise installed successfully"),)
    (os_L6).chdir(Str(".."),)
    return (su_L8).rmtree(Str("poise"),)

overwrite_L35 = mutate_operator(Bool(False))
if_tmp_func_3__()
(os_L6).mkdir(Str(erg_dir_L12),)
(os_L6).mkdir(Str(erg_bin_dir_L13),)
latest_version_L57 = if_tmp_func_12__()
(print)(((Str("version: ") + (str__)(latest_version_L57,)) + Str("")),)
filename_L75 = match_tmp_func_14__()
url_L79 = ((((Str("https://github.com/erg-lang/erg/releases/download/") + (str__)(latest_version_L57,)) + Str("/")) + (str__)(Str(filename_L75),)) + Str(""))
(print)(((Str("Downloading ") + (str__)(Str(url_L79),)) + Str(" ...")),)
stream_L83 = (
(urllib_L1).request
).urlopen(Str(url_L79),)
if_tmp_func_16__()
(print)(Str("erg installed successfully"),)
(install_poise__erg_proc___L16)()
(print)(((((Str("Please add `.erg` to your PATH by running `export PATH=$PATH:") + (str__)(Str(erg_bin_dir_L13),)) + Str("` and `export ERG_PATH=")) + (str__)(Str(erg_dir_L12),)) + Str("`")),) if (not (overwrite_L35)) else None
