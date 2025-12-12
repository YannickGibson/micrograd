

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a)
d += 3 * d + (b - a)
e = c - d
print(f'{e.data:.4f}') # prints 24.7041, the outcome of this forward pass
e.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of de/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of de/db
```
