import torch
from torch import optim
from torch import sparse
from unittest import TestCase
from torch.autograd import Variable
import functools
from torch_sparse_adamw import SparseAdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.Tensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


class TestOptim(TestCase):
    def _test_rosenbrock_sparse(self, constructor, scheduler_constructors=None, sparse_only=False):
        if scheduler_constructors is None:
            scheduler_constructors = []
        params_t = torch.Tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.Tensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.Tensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2])).to(dtype=v.dtype)
            with torch.no_grad():
                if sparse_grad:
                    params.grad = x
                else:
                    params.grad = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(rosenbrock(params))
                else:
                    scheduler.step()
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def test_sparse_adamw(self):
        self._test_rosenbrock_sparse(
            lambda params: SparseAdamW(params, lr=4e-2),
            [],
            True
        )
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
            SparseAdamW(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "SparseAdamW requires dense parameter tensors"):
            SparseAdamW([torch.zeros(3, layout=torch.sparse_coo)])
        with self.assertRaisesRegex(ValueError, "SparseAdamW requires dense parameter tensors"):
            SparseAdamW([{"params": [torch.zeros(3, layout=torch.sparse_coo)]}])
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            SparseAdamW(None, lr=1e-2, weight_decay=-1)

        # Test if the optimizer behaves the same on dense tensors as AdamW
        def to_sparse(x):
            """ converts dense tensor x to sparse format """
            x_typename = torch.typename(x).split('.')[-1]
            sparse_tensortype = getattr(torch.sparse, x_typename)

            indices = torch.nonzero(x)
            if len(indices.shape) == 0:  # if all elements are zeros
                return sparse_tensortype(*x.shape)
            indices = indices.t()
            values = x[tuple(indices[i] for i in range(indices.shape[0]))]
            return sparse_tensortype(indices, values, x.size())

        weight_data = torch.randn(10, 5)
        weight = Variable(weight_data, requires_grad=True)
        sparse_weight = Variable(weight_data.clone(), requires_grad=True)
        grad = torch.randn(10, 5)
        adamw = optim.AdamW([weight])
        optimizer = SparseAdamW([sparse_weight])

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        adamw.zero_grad()
        optimizer.zero_grad()

        for _ in range(3):
            grad = 1. + torch.randn(10, 5)
            weight._grad = grad
            sparse_weight._grad = to_sparse(grad)
            adamw.step()
            optimizer.step()

        adamw_state = next(iter(adamw.state.values()))
        for k, val in next(iter(optimizer.state.values())).items():
            torch.testing.assert_allclose(val, adamw_state[k])
        torch.testing.assert_allclose(weight.data, sparse_weight.data)


if __name__ == '__main__':
    import unittest
    unittest.main()
