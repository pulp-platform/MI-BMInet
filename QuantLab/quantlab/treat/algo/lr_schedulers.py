# Copyright (c) 2019 UniMoRe, Matteo Spallanzani

from torch.optim.lr_scheduler import _LRScheduler


class HandScheduler(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        self.schedule = schedule
        super(HandScheduler, self).__init__(optimizer, last_epoch=last_epoch)
        self.step(last_epoch + 1)

    def get_lr(self):
        return [base_lr * self.schedule[str(self.last_epoch)] for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if str(self.last_epoch) in self.schedule.keys():
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
