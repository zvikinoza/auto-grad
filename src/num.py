from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar('T')

class IValue(ABC): 
    def __init__(self, dat: T, children, operation) -> None:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, o):
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, o):
        raise NotImplementedError

    @abstractmethod
    def __pow__(self, o):
        raise NotImplementedError

    @abstractmethod
    def relu(self):
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        raise NotImplementedError

    @property
    def data(self):
        return self.__dat

    @property
    def grad(self):
        return self.__grad

    def __neg__(self): 
        return self * -1

    def __radd__(self, o):
        return self + o

    def __sub__(self, o): 
        return self + (-o)

    def __rsub__(self, o): 
        return o + (-self)

    def __rmul__(self, o):
        return self * o

    def __truediv__(self, o):
        return self * o ** -1

    def __rtruediv__(self, o):
        return o * self ** -1

    def __repr__(self):
        return f"Value(dat={self.__dat}, grad={self.__grad})"


def __wrap_if_needed(dat):
    if isinstance(dat, Value):
        return dat
    return Value(dat)




class Value(IValue):
    def __init__(self, dat, children: tuple[IValue] = (), operation=''):
        self.__dat = dat
        self.__grad = 0
        self.__backward = lambda: None
        self.__prev = set(children)
        self.__operation = operation 

    def __add__(self, o):
        o = __wrap_if_needed(o)
        out = Value(self.__dat + o.__dat, (self, o), '+')

        def __backward():
            self.__grad += out.__grad
            o.__grad += out.__grad
        out.__backward = __backward

        return out

    def __mul__(self, o):
        o = __wrap_if_needed(o)
        out = Value(self.__dat * o.__dat, (self, o), '*')

        def __backward():
            self.__grad += out.__grad * o.__dat
            o.__grad += out.__grad * self.__dat
        out.__backward = __backward

        return out

    def __pow__(self, o):
        out = Value(self.__dat ** o, (self,), f' ** {o}')

        def __backward():
            self.__grad += out.__grad * (o * self.__dat ** (o-1))
        out.__backward = __backward

        return out

    def relu(self):
        out = Value(max(0, self.__dat), (self,), 'relu')

        def __backward():
            self.__grad += out.__grad * (out.__dat > 0)
        out.__backward = __backward

        return out

    def backward(self):
        topo, visited = [], set()
        def topological_dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v.__prev:
                    topological_dfs(child)
                topo.append(v)
        
        topological_dfs(self)
        
        self.__grad = 1
        for v in topo[::-1]:
            v.__backward()
