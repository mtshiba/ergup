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
    obj.__enter__()
    body(e)
    obj.__exit__()

def discard__(obj):
    pass

def then__(x, f):
    if x == None or x == NotImplemented:
        return x
    else:
        return f(x)
# from typing import TypeVar, Union, _SpecialForm, _type_check

class Error:
    def __init__(self, message):
        self.message = message

# T = TypeVar("T")
# @_SpecialForm
def Result(self, parameters):
    """Result type.

    Result[T] is equivalent to Union[T, Error].
    """
    # arg = _type_check(parameters, f"{self} requires a single type.")
    return [arg, Error]

def is_ok(obj) -> bool:
    return not isinstance(obj, Error)



class Int(int):
    def try_new(i): # -> Result[Nat]
        if isinstance(i, int):
            Int(i)
        else:
            Error("not an integer")
    def succ(self):
        return Int(self + 1)
    def pred(self):
        return Int(self - 1)
    def mutate(self):
        return IntMut(self)
    def __add__(self, other):
        return then__(int.__add__(self, other), Int)
    def __radd__(self, other):
        return then__(int.__add__(other, self), Int)
    def __sub__(self, other):
        return then__(int.__sub__(self, other), Int)
    def __rsub__(self, other):
        return then__(int.__sub__(other, self), Int)
    def __mul__(self, other):
        return then__(int.__mul__(self, other), Int)
    def __rmul__(self, other):
        return then__(int.__mul__(other, self), Int)
    def __div__(self, other):
        return then__(int.__div__(self, other), Int)
    def __rdiv__(self, other):
        return then__(int.__div__(other, self), Int)
    def __floordiv__(self, other):
        return then__(int.__floordiv__(self, other), Int)
    def __rfloordiv__(self, other):
        return then__(int.__floordiv__(other, self), Int)
    def __pow__(self, other):
        return then__(int.__pow__(self, other), Int)
    def __rpow__(self, other):
        return then__(int.__pow__(other, self), Int)

class IntMut(): # inherits Int
    value: Int

    def __init__(self, i):
        self.value = Int(i)
    def __repr__(self):
        return self.value.__repr__()
    def __eq__(self, other):
        if isinstance(other, Int):
            return self.value == other
        else:
            return self.value == other.value
    def __ne__(self, other):
        if isinstance(other, Int):
            return self.value != other
        else:
            return self.value != other.value
    def __le__(self, other):
        if isinstance(other, Int):
            return self.value <= other
        else:
            return self.value <= other.value
    def __ge__(self, other):
        if isinstance(other, Int):
            return self.value >= other
        else:
            return self.value >= other.value
    def __lt__(self, other):
        if isinstance(other, Int):
            return self.value < other
        else:
            return self.value < other.value
    def __gt__(self, other):
        if isinstance(other, Int):
            return self.value > other
        else:
            return self.value > other.value
    def __add__(self, other):
        if isinstance(other, Int):
            return IntMut(self.value + other)
        else:
            return IntMut(self.value + other.value)
    def __sub__(self, other):
        if isinstance(other, Int):
            return IntMut(self.value - other)
        else:
            return IntMut(self.value - other.value)
    def __mul__(self, other):
        if isinstance(other, Int):
            return IntMut(self.value * other)
        else:
            return IntMut(self.value * other.value)
    def __floordiv__(self, other):
        if isinstance(other, Int):
            return IntMut(self.value // other)
        else:
            return IntMut(self.value // other.value)
    def __pow__(self, other):
        if isinstance(other, Int):
            return IntMut(self.value ** other)
        else:
            return IntMut(self.value ** other.value)
    def inc(self, i=1):
        self.value = Int(self.value + i)
    def dec(self, i=1):
        self.value = Int(self.value - i)
    def succ(self):
        return self.value.succ()
    def pred(self):
        return self.value.pred()


 # don't unify with the above line


class Nat(Int):
    def try_new(i): # -> Result[Nat]
        if i >= 0:
            return Nat(i)
        else:
            return Error("Nat can't be negative")

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

class NatMut(IntMut): # and Nat
    value: Nat

    def __init__(self, n: Nat):
        self.value = n
    def __repr__(self):
        return self.value.__repr__()
    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        else:
            return self.value == other.value
    def __ne__(self, other):
        if isinstance(other, int):
            return self.value != other
        else:
            return self.value != other.value
    def __le__(self, other):
        if isinstance(other, int):
            return self.value <= other
        else:
            return self.value <= other.value
    def __ge__(self, other):
        if isinstance(other, int):
            return self.value >= other
        else:
            return self.value >= other.value
    def __lt__(self, other):
        if isinstance(other, int):
            return self.value < other
        else:
            return self.value < other.value
    def __gt__(self, other):
        if isinstance(other, int):
            return self.value > other
        else:
            return self.value > other.value
    def __add__(self, other):
        if isinstance(other, Nat):
            return NatMut(self.value + other)
        else:
            return NatMut(self.value + other.value)
    def __radd__(self, other):
        if isinstance(other, Nat):
            return Nat(other + self.value)
        else:
            return Nat(other.value + self.value)
    def __mul__(self, other):
        if isinstance(other, Nat):
            return NatMut(self.value * other)
        else:
            return NatMut(self.value * other.value)
    def __rmul__(self, other):
        if isinstance(other, Nat):
            return Nat(other * self.value)
        else:
            return Nat(other.value * self.value)
    def __pow__(self, other):
        if isinstance(other, Nat):
            return NatMut(self.value ** other)
        else:
            return NatMut(self.value ** other.value)
    def try_new(i): # -> Result[Nat]
        if i >= 0:
            return NatMut(i)
        else:
            return Error("Nat can't be negative")

    def times(self, f):
        for _ in range(self.value):
            f()



class Bool(Nat):
    def try_new(b: bool): # -> Result[Nat]
        if b == True or b == False:
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




class Str(str):
    def __instancecheck__(cls, obj):
        return isinstance(obj, str)
    def try_new(s: str): # -> Result[Nat]
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
    def __add__(self, other):
        return then__(str.__add__(self, other), Str)
    def __radd__(self, other):
        return then__(str.__add__(other, self), Str)
    def __mul__(self, other):
        return then__(str.__mul__(self, other), Str)
    def __mod__(self, other):
        return then__(str.__mod__(other, self), Str)

class StrMut(): # Inherits Str
    value: Str

    def __init__(self, s: str):
        self.value = s
    def __repr__(self):
        return self.value.__repr__()
    def __eq__(self, other):
        if isinstance(other, Str):
            return self.value == other
        else:
            return self.value == other.value
    def __ne__(self, other):
        if isinstance(other, Str):
            return self.value != other
        else:
            return self.value != other.value
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
        self.value = self.value[:idx] + self.value[idx+1:]
        return char
    def insert(self, idx: int, s: str):
        self.value = self.value[:idx] + s + self.value[idx:]
class Array(list):
    def dedup(self):
        return Array(list(set(self)))
    def dedup_by(self, f):
        return Array(list(set(map(f, self))))
    def push(self, value):
        self.append(value)
        return self
urllib__ = __import__("urllib.request")
# from typing import TypeVar, Union, _SpecialForm, _type_check

class Error:
    def __init__(self, message):
        self.message = message

# T = TypeVar("T")
# @_SpecialForm
def Result(self, parameters):
    """Result type.

    Result[T] is equivalent to Union[T, Error].
    """
    # arg = _type_check(parameters, f"{self} requires a single type.")
    return [arg, Error]

def is_ok(obj) -> bool:
    return not isinstance(obj, Error)



# from collections.abc import Iterable, Sequence, Iterator, Container

class Range:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __contains__(self, item):
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

# represents `start<..<end`
class OpenRange(Range):
    def __contains__(self, item):
        return self.start < item < self.end

# represents `start..end`
class ClosedRange(Range):
    def __contains__(self, item):
        return self.start <= item <= self.end

class RangeIterator:
    def __init__(self, rng):
        self.rng = rng
        self.needle = self.rng.start
        if issubclass(Nat, type(self.rng.start)):
            if not(self.needle in self.rng):
                self.needle += 1
        elif issubclass(Str, type(self.rng.start)):
            if not(self.needle in self.rng):
                self.needle = chr(ord(self.needle) + 1)
        else:
            if not(self.needle in self.rng):
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



def in_operator(x, y):
    if type(y) == type:
        if isinstance(x, y):
            return True
        elif is_ok(y.try_new(x)):
            return True
        # TODO: trait check
        return False
    elif (issubclass(type(y), list) or issubclass(type(y), set)) \
        and (type(y[0]) == type or issubclass(type(y[0]), Range)):
        # FIXME:
        type_check = in_operator(x[0], y[0])
        len_check = len(x) == len(y)
        return type_check and len_check
    elif issubclass(type(y), dict) and issubclass(type(next(iter(y.keys()))), type):
        # TODO:
        type_check = True # in_operator(x[next(iter(x.keys()))], next(iter(y.keys())))
        len_check = len(x) >= len(y)
        return type_check and len_check
    else:
        return x in y
def match_tmp_func_1__():
    match (sys__).platform:
        case "darwin":
            match_tmp_0__ = Str("erg-x86_64-apple-darwin.tar.gz")
        case "win32":
            match_tmp_0__ = Str("erg-x86_64-pc-windows-msvc.zip")
        case _:
            match_tmp_0__ = Str("erg-x86_64-unknown-linux-gnu.tar.gz")
    return match_tmp_0__
os__ = __import__("os.path")
urllib__ = __import__("urllib.request")
def if_tmp_func_3__():
    if ((sys__).platform == Str("win32")):
        (print)(((Str("extracting ") + filename__) + Str(" ...")),)
        bytesio__ = (io__).BytesIO((stream__).read(),)
        zipfile__ = (zf__).ZipFile(bytesio__,)
        (zipfile__).extractall((homedir__ + Str("/.erg/tmp")),)
        (zipfile__).close()
        (su__).move((homedir__ + Str("/.erg/tmp/erg.exe")),(homedir__ + Str("/.erg/bin/erg.exe")),)
        (su__).move((homedir__ + Str("/.erg/tmp/lib")),(homedir__ + Str("/.erg/lib")),)
        if_tmp_2__ = (su__).rmtree((homedir__ + Str("/.erg/tmp")),)
    else:
        (print)(((Str("extracting ") + filename__) + Str(" ...")),)
        tarfile__ = (tf__).open(fileobj=stream__,mode=Str("r|gz"),)
        (tarfile__).extractall((homedir__ + Str("/.erg/tmp")),)
        (tarfile__).close()
        (su__).move((homedir__ + Str("/.erg/tmp/erg")),(homedir__ + Str("/.erg/bin/erg")),)
        (su__).move((homedir__ + Str("/.erg/tmp/lib")),(homedir__ + Str("/.erg/lib")),)
        if_tmp_2__ = (su__).rmtree((homedir__ + Str("/.erg/tmp")),)
    return if_tmp_2__
urllib__ = (__import__)(Str("urllib"),)
tf__ = (__import__)(Str("tarfile"),)
zf__ = (__import__)(Str("zipfile"),)
json__ = (__import__)(Str("json"),)
sys__ = (__import__)(Str("sys"),)
os__ = (__import__)(Str("os"),)
io__ = (__import__)(Str("io"),)
su__ = (__import__)(Str("shutil"),)
latest_url__ = Str("https://api.github.com/repos/erg-lang/erg/releases/latest")
_stream__ = (
(urllib__).request
).urlopen(latest_url__,)
s__ = ((_stream__).read()).decode()
jdata__ = (json__).loads(s__,)
assert in_operator(jdata__,{(Str): (Str),})
latest_version__ = (jdata__).__getitem__(Str("tag_name"),)
(print)((Str("version: ") + latest_version__),)
filename__ = match_tmp_func_1__()
url__ = (((Str("https://github.com/erg-lang/erg/releases/download/") + latest_version__) + Str("/")) + filename__)
(print)(((Str("downloading ") + url__) + Str(" ...")),)
homedir__ = (
    (os__).path
    ).expanduser(Str("~"),)
(os__).mkdir((homedir__ + Str("/.erg")),)
(os__).mkdir((homedir__ + Str("/.erg/bin")),)
stream__ = (
    (urllib__).request
    ).urlopen(url__,)
if_tmp_func_3__()
(print)(((((Str("please add `.erg` to your PATH by running `export PATH=$PATH:") + homedir__) + Str("/.erg/bin` and `export ERG_PATH=")) + homedir__) + Str("/.erg`")),)
