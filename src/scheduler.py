import math

class Scheduler: 
    """Linear warmup and decay scheduler for learning rate."""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return (self.current_step / self.warmup_steps) * 1e-3
        else:
            return max(1e-5, 1e-3 * (1 - (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
