# Implementing a Node Class for Automatic Differentiation

This project implements a `Node` class to represent values in a **computational graph**, enabling **automatic differentiation**. Each operation on `Node` objects creates new nodes that store both the result and the relationship to their inputs. This setup allows the computation of gradients through **reverse-mode differentiation** (commonly known as backpropagation).

---

## Key Points:

- Each Node stores its value, gradient, operation, and references to children nodes
- Overloads basic arithmetic operations: `+`, `-`, `*`, `/`, and `**`  
- Supports scalar and `Node` combinations (e.g., `Node + scalar`)
- Automatically tracks dependencies in a computational graph
- Computes gradients via a `.backward()` method using the chain rule
---

## `backward()` Method

The `backward()` function computes the gradient of an output node with respect to all variables it depends on, by recursively applying the **chain rule** through the computational graph.

It handles:
- Addition/Subtraction (pass-through gradients)
- Multiplication (product rule or 'u.v' rule)
- Division (quotient rule or 'u/v' rule)
- Exponentiation (power rule)

---


