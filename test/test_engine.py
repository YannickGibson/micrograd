import torch
from micrograd.engine import Value

def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z + z * x
    h = (z * z)
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z + z * x
    h = (z * z)
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a)
    d += 3 * d + (b - a)
    e = c - d
    e.backward()
    amg, bmg, gmg = a, b, e

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a)
    d = d + 3 * d + (b - a)
    e = c - d
    e.backward()
    apt, bpt, gpt = a, b, e

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
