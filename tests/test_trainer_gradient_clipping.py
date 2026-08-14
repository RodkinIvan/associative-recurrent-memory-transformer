import unittest
from types import SimpleNamespace

import torch

from lm_experiments_tools.trainer import Trainer


class _Accelerator:
    def __init__(self):
        self.value_calls = 0
        self.norm_calls = 0

    def clip_grad_value_(self, parameters, clip_value):
        self.value_calls += 1
        torch.nn.utils.clip_grad_value_(parameters, clip_value)

    def clip_grad_norm_(self, parameters, max_norm):
        self.norm_calls += 1
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)


class GradientClippingTest(unittest.TestCase):
    def test_clip_grad_value_is_elementwise(self):
        trainer = object.__new__(Trainer)
        trainer.model = torch.nn.Linear(2, 2)
        for parameter in trainer.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer.args = SimpleNamespace(clip_grad_value=0.1, clip_grad_norm=None)
        trainer.accelerator = _Accelerator()

        trainer._clip_gradients()

        self.assertEqual(trainer.accelerator.value_calls, 1)
        self.assertEqual(trainer.accelerator.norm_calls, 0)
        for parameter in trainer.model.parameters():
            torch.testing.assert_close(
                parameter.grad, torch.full_like(parameter, 0.1), atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
