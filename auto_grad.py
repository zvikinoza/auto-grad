class Node():
    """Base class for evry node in computational graph"""

    def __init__(self, name):
        self.name = name
        self._sub_nodes = []
        self._const_val = 0

    def _add_sub_nodes(self, *nodes):
        """ add nodes in computational graph under self
        Args:
            nodes (iterable): iterable of Node() s
        """
        for node in nodes:
            self.__add_sub_node(node)

    def __add_sub_node(self, node):
        self._sub_nodes.append(node)

    def _is_atomic(self):
        return not self._sub_nodes

    def __str__(self):
        return self.name

    __repr__ = __str__

    # Abstract methods

    def __add__(self, other):
        raise NotImplemented

    def __sub__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        raise NotImplemented


class Var(Node):
    def __init__(self, name):
        super().__init__(name)

    @classmethod
    def __init__from_var(cls, other):
        var = cls(other.name)
        var._const_val = other._const_val
        var._add_sub_nodes(*other._sub_nodes)
        return var

    def eval(self, values):
        if self._is_atomic():
            return values.get(self.name, 0) + self._const_val
        # Var always has only one sub node which always is
        # Op (operation ) node or Op's subclass
        return self._sub_nodes[0].eval(values) + self._const_val

    def __add__(self, other):
        if isinstance(other, Var):
            return self.__add_var_to_var(other)
        return self.__add_num_to_var(other)

    def __sub__(self, other):
        if isinstance(other, Var):
            return self.__sub_var_to_var(other)
        return self.__add_num_to_var(-other)

    def __rsub__(self, other):
        if isinstance(other, Var):
            return other.__sub__(self)
        neg_var = self * -1
        return neg_var + other

    def __mul__(self, other):
        if isinstance(other, Var):
            return self.__mul_var_to_var(other)
        return self.__mul_num_to_var(other)

    __radd__ = __add__
    __rmul__ = __mul__

    def _join_vars_and_ops(self, op_node, res_node, f_node, s_node):
        op_node._add_sub_nodes(f_node, s_node)
        res_node._add_sub_nodes(op_node)
        return res_node

    def __mul_var_to_var(self, other):
        op_node = Mul('mul_op')
        mul_node = Var('mul_var')
        return self._join_vars_and_ops(op_node, mul_node, self, other)

    def __add_var_to_var(self, other):
        op_node = Add('add_op')
        sum_node = Var('sum_var')
        return self._join_vars_and_ops(op_node, sum_node, self, other)

    def __mul_num_to_var(self, other):
        op_node = Mul('mul_op')
        mul_node = Var('mul_var')
        multiplicator = Var('multiplicator')
        multiplicator._const_val += other
        return self._join_vars_and_ops(op_node, mul_node, self, multiplicator)

    def __add_num_to_var(self, other):
        self._const_val += other
        return self

    def __sub_var_to_var(self, other):
        op_node = Sub('sub_op')
        sub_node = Var('sub_var')
        return self._join_vars_and_ops(op_node, sub_node, self, other)


class Op(Node):
    def __init__(self, name):
        super().__init__(name)

    def eval(self, values):
        raise NotImplemented


class Add(Op):
    def __init__(self, name):
        super().__init__(name)

    def eval(self, values):
        sub_vars_values = [node.eval(values) for node in self._sub_nodes]
        return sum(sub_vars_values)


class Mul(Op):
    def __init__(self, name):
        super().__init__(name)

    def eval(self, values):
        sub_vars_values = [node.eval(values) for node in self._sub_nodes]
        res = sub_vars_values[0]
        for val in sub_vars_values[1:]:
            res *= val
        return res


class Sub(Op):
    def __init__(self, name):
        super().__init__(name)

    def eval(self, values):
        sub_vars_values = [node.eval(values) for node in self._sub_nodes]
        return sub_vars_values[0] - sub_vars_values[1]
