"""
CPU-offloaded optimizer for single-GPU training.

Keeps optimizer states (Adam m/v) on CPU, freeing ~5-7 GB of VRAM.
States are only needed during optimizer.step(), not forward/backward.

Usage:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer = CPUOffloadOptimizer(optimizer)
    # Use normally — .step(), .zero_grad(), .state_dict() all work.
"""

import torch


class CPUOffloadOptimizer:
    """Wraps a GPU optimizer to keep states on CPU.

    On .step(): copies gradients to CPU, steps on CPU, copies params back to GPU.
    On .zero_grad(): delegates to the underlying optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._gpu_params = []
        self._cpu_params = []
        self._param_map = {}  # gpu_param -> cpu_param

        # Create CPU mirror of all parameters
        for group in optimizer.param_groups:
            gpu_params = []
            cpu_params = []
            for p in group["params"]:
                cpu_p = p.detach().float().cpu()
                cpu_p.requires_grad = True
                self._param_map[p] = cpu_p
                gpu_params.append(p)
                cpu_params.append(cpu_p)
            self._gpu_params.append(gpu_params)
            self._cpu_params.append(cpu_params)
            # Point optimizer at CPU params
            group["params"] = cpu_params

    @torch.no_grad()
    def step(self):
        # Copy gradients GPU → CPU
        for gpu_group, cpu_group in zip(self._gpu_params, self._cpu_params):
            for gpu_p, cpu_p in zip(gpu_group, cpu_group):
                if gpu_p.grad is not None:
                    cpu_p.grad = gpu_p.grad.float().cpu()

        # Step on CPU
        self.optimizer.step()

        # Copy updated params CPU → GPU
        for gpu_group, cpu_group in zip(self._gpu_params, self._cpu_params):
            for gpu_p, cpu_p in zip(gpu_group, cpu_group):
                gpu_p.data.copy_(cpu_p.data)

    def zero_grad(self, set_to_none=False):
        # Zero GPU grads (where backward accumulates)
        for gpu_group in self._gpu_params:
            for p in gpu_group:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.zero_()
        # Zero CPU grads
        for cpu_group in self._cpu_params:
            for p in cpu_group:
                if set_to_none:
                    p.grad = None
                elif p.grad is not None:
                    p.grad.zero_()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
