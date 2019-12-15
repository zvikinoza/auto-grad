import random
from typing import List, Iterable, Union, Any
from num import IValue, Value
from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def parameters(self) -> Iterable[IValue]:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

class Neuron(Module):
    def __init__(self, n_in: int, non_linear: bool = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b = Value(0)
        self.non_linear = non_linear

    def __call__(self, x: Iterable[IValue]) -> IValue:
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        if self.non_linear:
            return act.relu()
        return act

    def parameters(self) -> Iterable[IValue]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.non_linear else 'Lin'} Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, n_in: int, n_out: int, **kwargs) -> None:
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x: IValue) -> Union[IValue, List[IValue]]:
        if len(self.neurons) == 1:
            return self.neurons[0](x)
        return [n(x) for n in self.neurons]

    def parameters(self) -> Any:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class NeuralNet(Module):
    def __init__(self, n_in, n_out: List[int]) -> None:
        sz = [n_in] + n_out
        self.layers = [Layer(sz[i], sz[i+1], non_linear = (i != len(n_out)-1)) for i in range(len(n_out))]

    def __call__(self, x) -> Union[IValue, List[IValue]]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[IValue]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"NeuralNet of [{', '.join(str(layer) for layer in self.layers)}]"
