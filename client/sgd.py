import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from loguru import logger
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from model import Linear, MLP
from dataset import CustomDataset
import client.base


class Worker(client.base.Worker):
    def init(self):
        self.n = self.train_data.shape[1]
        self._model = Linear if self.args.model == 'Linear' else MLP
        self.model = self._model(self.n, self.C).double().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.train_dataset = CustomDataset(torch.from_numpy(self.train_data))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.test_dataset = CustomDataset(torch.from_numpy(self.test_data))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
        if self.args.use_DP:
            self.privacy_engine = PrivacyEngine(accountant='rdp')
            self.model, self.optimizer, self.train_dataloader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dataloader,
                epochs=self.args.epoch,
                target_epsilon=self.args.target_epsilon,
                target_delta=self.args.DP_delta,
                max_grad_norm=self.args.max_per_sample_clip_norm,
                poisson_sampling=False,
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                # alphas=list(range(2, 5000)) # UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound
            )
            logger.info(f"Using sigma={self.optimizer.noise_multiplier} and C={self.args.max_per_sample_clip_norm}, target epsilon={self.args.target_epsilon}")

    def prepare_dataloader_iter(self):
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.test_dataloader_iter = iter(self.test_dataloader)

    def prepare_batch(self, mode='train'):
        if mode == 'train':
            self.batch = next(self.train_dataloader_iter).to(self.device)
        elif mode == 'test':
            self.batch = next(self.test_dataloader_iter).to(self.device)
    
    def one_step_forward(self, mode='train'):
        if mode == 'train':
            self.model.train()
            self.local_tmp_embedding = self.model(self.batch)
            self.embedding = torch.autograd.Variable(self.local_tmp_embedding.data, requires_grad=True)  
        elif mode == 'test':
            self.model.eval()
            with torch.no_grad():
                self.embedding = self.model(self.batch)

    def one_step_backward(self):
        self.local_tmp_embedding.backward(self.embedding.grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_privacy_budget(self):
        if not self.args.use_DP:
            return None
        return self.privacy_engine.get_epsilon(self.args.DP_delta)