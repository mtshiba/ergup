urllib__ = __import__("urllib.request")
from typing import TypeVar, Union, _SpecialForm, _type_check

class Error:
    def __init__(self, message):
        self.message = message

T = TypeVar("T")
@_SpecialForm
def Result(self, parameters):
    """Result type.

    Result[T] is equivalent to Union[T, Error].
    """
    arg = _type_check(parameters, f"{self} requires a single type.")
    return Union[arg, Error]

def is_ok(obj: Result[T]) -> bool:
    return not isinstance(obj, Error)



from collections.abc import Iterable, Sequence, Iterator, Container

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

Sequence.register(Range)
Container.register(Range)
Iterable.register(Range)

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
                self.needle = self.needle.incremented()

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
                self.needle = self.needle.incremented()
                return result
        raise StopIteration

Iterator.register(RangeIterator)



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


class Nat(int):
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

class StrMut(Str):
    def try_new(s: str):
        if isinstance(s, str):
            return StrMut(s)
        else:
            return Error("Str! can't be other than str")
    def clear(self):
        self = ""
    def pop(self):
        if len(self) > 0:
            last = self[-1]
            self = self[:-1]
            return last
        else:
            return Error("Can't pop from empty `Str!`")
    def push(self, c: str):
        self += c
    def remove(self, idx: int):
        char = self[idx]
        self = self[:idx] + self[idx+1:]
        return char
    def insert(self, idx: int, c: str):
        self = self[:idx] + c + self[idx:]
class Array(list):
    def dedup(self):
        return Array(list(set(self)))
    def dedup_by(self, f):
        return Array(list(set(map(f, self))))
    def push(self, value):
        self.append(value)
        return self
def match_tmp_func_1__():
    match (sys__).platform:
        case "darwin":
            match_tmp_0__ = "erg-x86_64-apple-darwin.tar.gz"
        case "win32":
            match_tmp_0__ = "erg-x86_64-pc-windows-msvc.zip"
        case _:
            match_tmp_0__ = "erg-x86_64-unknown-linux-gnu.tar.gz"
    return match_tmp_0__
os__ = __import__("os.path")
urllib__ = __import__("urllib.request")
def if_tmp_func_3__():
    if ((sys__).platform == "win32"):
        (print)((("extracting " + filename__) + " ..."),)
        bytesio__ = (io__).BytesIO((stream__).read(),)
        zipfile__ = (zf__).ZipFile(bytesio__,)
        (zipfile__).extractall()
        (zipfile__).close()
        (su__).move("erg.exe",(homedir__ + "/.erg/bin/erg.exe"),)
        if_tmp_2__ = (su__).move("lib",(homedir__ + "/.erg/lib"),)
    else:
        (print)((("extracting " + filename__) + " ..."),)
        tarfile__ = (tf__).open(fileobj__=stream__,mode__="r|gz",)
        (tarfile__).extractall()
        (tarfile__).close()
        (su__).move("erg",(homedir__ + "/.erg/bin/erg"),)
        if_tmp_2__ = (su__).move("lib",(homedir__ + "/.erg/lib"),)
    return if_tmp_2__
urllib__ = (__import__)("urllib",)
tf__ = (__import__)("tarfile",)
zf__ = (__import__)("zipfile",)
json__ = (__import__)("json",)
sys__ = (__import__)("sys",)
os__ = (__import__)("os",)
io__ = (__import__)("io",)
su__ = (__import__)("shutil",)
latest_url__ = "https://api.github.com/repos/erg-lang/erg/releases/latest"
_stream__ = (
(urllib__).request
).urlopen(latest_url__,)
s__ = ((_stream__).read()).decode()
jdata__ = (json__).loads(s__,)
assert in_operator(jdata__,{(Str): (Str),})
latest_version__ = (jdata__).__getitem__("tag_name",)
(print)(("version: " + latest_version__),)
filename__ = match_tmp_func_1__()
url__ = ((("https://github.com/erg-lang/erg/releases/download/" + latest_version__) + "/") + filename__)
(print)((("downloading " + url__) + " ..."),)
homedir__ = (
    (os__).path
    ).expanduser("~",)
(os__).mkdir((homedir__ + "/.erg"),)
(os__).mkdir((homedir__ + "/.erg/bin"),)
stream__ = (
    (urllib__).request
    ).urlopen(url__,)
if_tmp_func_3__()
(print)((((("please add `.erg` to your PATH by running `export PATH=$PATH:" + homedir__) + "/.erg/bin` and `export ERG_PATH=") + homedir__) + "/.erg`"),)
