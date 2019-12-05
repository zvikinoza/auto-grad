from computation_graph import ComputationGraph

operations = ['+', '-', '*', '/']


def print_symbolic_library():
    print(f'You can only use {operations} to describe equation.')


def get_equation_and_variables():
    equation = input('Input equation: ')
    variables = input('Input space separated variables wrt to derivate: ')
    variables = variables.split(' ')
    return equation, variables


def main():
    print_symbolic_library()
    equation, variables = get_equation_and_variables()
    graph = ComputationGraph.from_raw_equation(equation, variables)
    gradient = graph.calculate_gradient()
    print(gradient)

if __name__ == '__main__':
    main()
