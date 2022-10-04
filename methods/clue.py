import numpy as np
import torch
import torch.nn.functional as F
import copy

from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche.benchmarks.utils import AvalancheSubset, AvalancheConcatDataset, AvalancheDataset
from torch.utils.data import DataLoader

class DistPlugin(StrategyPlugin):

    def __init__(self, mem_size=200, device='cpu', storage_policy='random', alpha=0.1, beta=0, disalpha=1, temperature=2):
        super(DistPlugin).__init__()
        self.mem_size = mem_size
        self.storage_policy = storage_policy
        self.current_batch = 0
        self.current_epoch = 0
        self.current_task = 0
        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.buffer_group = {}
        self.buffer_size = []
        self.logits = None

        self.disalpha = disalpha
        self.temperature = temperature
        self.prev_model = None

        self.prev_classes = {'0': set()}

    def calculate_buffer_size(self):
        if not self.buffer_size:
            self.buffer_size = [self.mem_size]
        else:
            mean_size = int(self.mem_size / self.current_task)
            for i in range(self.current_task - 1):
                self.buffer_size[i] = mean_size
            self.buffer_size.append(self.mem_size - mean_size * (self.current_task - 1))

    def distillation_loss(self, out, prev_out):
        log_p = torch.log_softmax(out / self.temperature, dim=1)
        q = torch.softmax(prev_out / self.temperature, dim=1)
        res = torch.nn.functional.kl_div(log_p, q, reduction='batchmean')
        return res

    def penalty(self, out, x, disalpha):
        if self.prev_model is None:
            return 0
        else:
            with torch.no_grad():
                yp = self.prev_model(x)
                yc = out
            dist_loss = 0
            dist_loss += self.distillation_loss(yc, yp)
            return disalpha * dist_loss

    def before_training_exp(self, strategy, num_workers: int = 0, shuffle: bool = True, **kwargs):
        self.current_task += 1
        self.current_batch = 0
        self.current_epoch = 0
        print('Begin training task ', self.current_task)

    def before_backward(self, strategy, **kwargs):
        if self.current_task == 1:
            return

        buffers = AvalancheConcatDataset([])
        for i in range(len(self.buffer_group)):
            cat_data = AvalancheConcatDataset([buffers, self.buffer_group[i]])
            buffers = cat_data

        buffer_dataloader = DataLoader(buffers, batch_size=strategy.train_mb_size, shuffle=False)
        log = torch.tensor([]).to(self.device)
        cla = torch.tensor([]).to(self.device)

        for img, label, domain in buffer_dataloader:
            img = img.to(self.device)
            label = label.to(self.device)
            logits = strategy.model(img)
            if not log.shape:
                log = logits
            else:
                log = torch.cat((log, logits), 0)
            if not cla.shape:
                cla = label
            else:
                cla = torch.cat((cla, label), 0)

        log_penalty = self.alpha * F.mse_loss(log, self.logits)
        cla_penalty = self.beta * F.cross_entropy(log, cla.long())
        #print(strategy.loss)
        strategy.loss += log_penalty
        strategy.loss += cla_penalty
        #print(strategy.loss)

        dis_penalty = self.penalty(strategy.mb_output, strategy.mb_x, self.disalpha)
        strategy.loss += dis_penalty
        #print(strategy.loss)


    def after_training_iteration(self, strategy: 'BaseStrategy', **kwargs):
        self.current_batch += 1
        if self.current_batch % 1 == 10:
            print('Task ' + str(self.current_task) + '; epoch ' + str(self.current_epoch) + '; batch ' + str(
                self.current_batch) + 'trained')

    def after_training_epoch(self, strategy: 'BaseStrategy', **kwargs):
        self.current_epoch += 1
        self.current_batch = 0

    def after_training_exp(self, strategy: 'BaseStrategy', **kwargs):
        self.calculate_buffer_size()
        for i in range(self.current_task - 1):
            origin_buffer = self.buffer_group[i]
            trimmed_idx = torch.tensor(np.random.choice(len(origin_buffer), self.buffer_size[i], replace=False))
            self.buffer_group[i] = AvalancheSubset(origin_buffer, trimmed_idx)

        current_idx = np.random.choice(len(strategy.experience.dataset), self.buffer_size[self.current_task - 1],
                                       replace=False)
        self.buffer_group[self.current_task - 1] = AvalancheSubset(strategy.experience.dataset, current_idx)

        buffers = AvalancheConcatDataset([])
        for i in range(len(self.buffer_group)):
            cat_data = AvalancheConcatDataset([buffers, self.buffer_group[i]])
            buffers = cat_data
        temp_loader = DataLoader(buffers, batch_size=strategy.train_mb_size, shuffle=False)
        temp_logits = torch.tensor([]).to(self.device)
        with torch.no_grad():
            for img, label, domain in temp_loader:
                img = img.to(self.device)
                logits = strategy.model(img)
                if not temp_logits.shape:
                    temp_logits = logits
                else:
                    temp_logits = torch.cat((temp_logits, logits), 0)

        self.logits = temp_logits
        self.prev_model = copy.deepcopy(strategy.model)

