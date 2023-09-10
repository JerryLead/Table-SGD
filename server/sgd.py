import torch
import math
import server.base
import numpy as np


class Server(server.base.Server):
    def init(self):
        self.loss = torch.nn.CrossEntropyLoss() if self.task == 'classification' else torch.nn.MSELoss()
        self.G = self.aligned_G.copy()
        self.test_G = self.aligned_test_G.copy()
        for i, worker in enumerate(self.workers):
            self.G[i], self.test_G[i] = worker.align(self.G[i], self.test_G[i], train_idx=self.f[:, i + 1], test_idx=self.test_f[:, i + 1])
            worker.init()
            # worker.align(self.f[:, i + 1], self.test_f[:, i + 1])
            # worker.init()
        self.train_iterations = math.ceil(self.M / self.args.batch_size)
        self.test_iterations = math.ceil(self.test_M / self.args.batch_size)
        if isinstance(self.b, np.ndarray):
            self.b = torch.from_numpy(self.b)
        if isinstance(self.test_b, np.ndarray):
            self.test_b = torch.from_numpy(self.test_b)
        self.b, self.test_b = self.b.to(self.device), self.test_b.to(self.device)
    
    def train_per_epoch(self):
        self.total_loss = 0
        if self.task == 'classification':
            self.correct = 0
        for worker in self.workers:
            worker.prepare_dataloader_iter()
        for batch_id in range(self.train_iterations):
            sample_start_idx = batch_id * self.args.batch_size
            sample_end_idx = (batch_id + 1) * self.args.batch_size
            label = self.b[sample_start_idx : sample_end_idx]
            loss = self._process_batch(label)
            loss.backward()
            for worker in self.workers:
                worker.one_step_backward()

    def _process_batch(self, label, mode='train'):
        if self.task == 'regression':
            label = label.reshape(-1, 1).double()
        for worker in self.workers:
            worker.prepare_batch(mode)
            worker.one_step_forward(mode)
        h = torch.sum(torch.stack([self.workers[i].embedding for i in range(self.N)]), dim=0)
        loss = self.loss(h, label)
        if mode == 'train':
            self.total_loss += loss.item()* len(label) / self.M
        else:
            self.test_total_loss += loss.item() * len(label) / self.test_M
        if self.task == 'classification':
            _, predicted = torch.max(h.data, 1)
            if mode == 'train':
                self.correct += (predicted == label).sum().item()
            else:
                self.test_correct += (predicted == label).sum().item()
        return loss

    def validate(self, mode='train'):
        if mode == 'train':
            total_loss = self.total_loss
            if self.task == 'classification':
                correct, M = self.correct, self.M
        else:
            total_loss = self.test_total_loss
            if self.task == 'classification':
                correct, M = self.test_correct, self.test_M
        loss = total_loss
        if self.task == 'classification':
            acc = correct / M
            return loss, acc
        elif self.task == 'regression':
            rmse = math.sqrt(loss)
            return rmse

    def test(self):
        self.test_total_loss = 0
        if self.task == 'classification':
            self.test_correct = 0
        for batch_id in range(self.test_iterations):
            sample_start_idx = batch_id * self.args.batch_size
            sample_end_idx = (batch_id + 1) * self.args.batch_size
            label = self.test_b[sample_start_idx : sample_end_idx]
            self._process_batch(label, mode='test')
        return self.validate('test')