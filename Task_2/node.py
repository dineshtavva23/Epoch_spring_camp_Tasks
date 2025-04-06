import numpy as np

class Node:
    def __init__(self, value, op=None, children=[], op_args=None):
        self.value = np.array(value)  # Storing value as a NumPy array
        self.children = children   # List of children nodes 
        self.op = op    # Symbol of operation used
        self.op_args = op_args if op_args else []  # Arguments provided 
        self.grad = 0   # Gradient of node

    # Overloading addition
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(value=self.value + other.value, op='+', children=[self, other])

    # Overloading multiplication
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(value=self.value * other.value, op='*', children=[self, other])

    # Overloading subraction
    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(value=self.value - other.value, op='-', children=[self, other])

    # Overloading divison
    def __truediv__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(value=self.value / other.value, op='/', children=[self, other])

    # Overloading exponentiation operator
    def __pow__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return Node(value=(self.value) ** other.value, op='**', children=[self, other])

    # To print the information in node
    def __repr__(self):
        return f"Node(value={self.value}, operation={self.op})"

    # Function to perform backward differentiation.
    def backward(self, grad=1):
        self.grad += grad  # Updating the gradient

        if self.op == '+':
            # Gradient is passed as the same 
            self.children[0].backward(grad)
            self.children[1].backward(grad)

        elif self.op == '*':
            # Using multiplication rule with a constant i.e. d/dx(x*y)= y and d/dy(x*y) = x
            self.children[0].backward(self.children[1].value * grad)
            self.children[1].backward(self.children[0].value * grad)

        elif self.op == '-':
            # Similarly d/dx(x-y)=1 and d/dx(x-y) =-1
            self.children[0].backward(grad)
            self.children[1].backward(-1 * grad)

        elif self.op == '/':
            # Using u/v rule i.e. d/dx(x/y) = 1/y and d/dy(x/y) = -x/y^2
            self.children[0].backward(grad / self.children[1].value)
            self.children[1].backward(-1 * grad * self.children[0].value / (self.children[1].value ** 2))

        elif self.op == '**':
            a = self.children[0].value  # Base
            b = self.children[1].value  # Exponent

            # d/da(a^b) = b * (a^(b-1))
            self.children[0].backward(grad * b * (a ** (b - 1)))

            # d/db(a^b) = a^b * ln(a) : so a should greater than 0 as ln(a) term
            # is involved
            if a > 0:
                self.children[1].backward(grad * (a ** b) * np.log(a))

        

x = Node(4)
y = Node(9)
z = Node(2)

m = x + y
n = x ** y

print("m :  ",m)
print("n :  ", n)

print(f"Children of m :  {m.children}")
print(f"Children of n :  {n.children}")

a = Node(6)
b = Node(4)

c = (a * 2) + b
d = c ** 3
d.backward()


print(f"Gradient of d with respect to a: {a.grad}")
print(f"Gradient of d with respect to b: {b.grad}")
print(f"Gradient of d with respect to c: {c.grad}")