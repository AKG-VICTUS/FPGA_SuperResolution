import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1., max_lr=0.1, min_lr=0.001,
                 warmup_steps=0, gamma=1., last_epoch=-1):
        assert warmup_steps < first_cycle_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        self.optimizer = optimizer
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
        elif self.step_in_cycle < self.warmup_steps:
            return [self.min_lr + (self.max_lr - self.min_lr) *
                    self.step_in_cycle / self.warmup_steps for _ in self.base_lrs]
        else:
            cos_inner = math.pi * (self.step_in_cycle - self.warmup_steps) / (
                    self.cur_cycle_steps - self.warmup_steps)
            return [self.min_lr + (self.max_lr - self.min_lr) *
                    (1 + math.cos(cos_inner)) / 2 for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** self.cycle))
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                                     self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps *
                                                     (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = int(self.first_cycle_steps * (self.cycle_mult ** n))
                    self.max_lr = self.base_max_lr * (self.gamma ** n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

