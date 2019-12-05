from auto_grad import *

def test_identity():
    x = Var('x')
    y = x
    values = { 'x':10 }
    y_val = y.eval(values)

    assert y_val == 10, f'Value mismatch expected 10 got {y_val}'
    print('Pass')

def test_add_const():
    x = Var('x')
    y = x + 20
    values = { 'x':5 }
    y_val = y.eval(values)

    assert y_val == 25, f'Value mismatch expected 25 got {y_val}'
    print('Pass')

def test_radd_const():
    x = Var('x')
    y = 3 + x
    values = { 'x':5 }
    y_val = y.eval(values)

    assert y_val == 8, f'Value mismatch expected 8 got {y_val}'
    print('Pass')

def test_add_vars():
    x = Var('x')
    y = Var('y')
    z = x + y
    values = { 'x':5, 'y':20}
    z_val = z.eval(values)

    assert z_val == 25, f'Value mismatch expected 25 got {z_val}'
    print('Pass')

def test_sub_const():
    x = Var('x')
    y = x - 2
    values = { 'x':5 }
    y_val = y.eval(values)

    assert y_val == 3, f'Value mismatch expected 3 got {y_val}'
    print('Pass')

def test_rsub_const():
    x = Var('x')
    y = 2 - x
    values = { 'x':5 }
    y_val = y.eval(values)

    assert y_val == -3, f'Value mismatch expected -3 got {y_val}'
    print('Pass')

def test_sub_vars():
    x = Var('x')
    y = Var('y')
    z = x - y
    values = { 'x':20, 'y':5}
    z_val = z.eval(values)

    assert z_val == 15, f'Value mismatch expected 25 got {z_val}'
    print('Pass')

def test_mult_const():
    x = Var('x')
    y = x * 5
    values = { 'x':10 }
    y_val = y.eval(values)

    assert y_val == 50, f'Value mismatch expected 50 got {y_val}'
    print('Pass')

def test_mult_vars():
    x = Var('x')
    y = Var('y')
    z = x * y
    values = { 'x':5, 'y':20}
    z_val = z.eval(values)

    assert z_val == 100, f'Value mismatch expected 100 got {z_val}'
    print('Pass')

def test_full_graph():
    b = Var('b')
    c = Var('c')
    d = Var('d')
    e = Var('e')
    a = b + c * d - e
    values = { 'b':3, 'c':.2, 'd':5, 'e':7 }
    a_val = a.eval(values)

    assert a_val == -3, f'Value mismatch expected -3 got {a_val}'
    print('Pass')

def test_construction_and_evaluation():
    test_identity()

    test_add_const()
    test_radd_const()
    test_add_vars()

    test_sub_const()
    test_rsub_const()
    test_sub_vars()

    test_mult_const()
    test_mult_vars()
    test_full_graph()

def test_gradient():
    pass

def main():
    test_construction_and_evaluation()
    test_gradient()

if __name__ == '__main__':
    main()
