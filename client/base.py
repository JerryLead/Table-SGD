import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from loguru import logger
import os
import abc


class Worker:
    def __init__(self, table_name):
        self.table_name = table_name

    def load_data(self, dataset, data_path):
        self.dataset = dataset
        self.C = self.dataset.num_class
        logger.info(f'Begin loading {self.table_name}')
        self.train_df = pd.read_csv(os.path.join(data_path, 'train', self.table_name + '.csv'))
        self.test_df = pd.read_csv(os.path.join(data_path, 'test', self.table_name + '.csv'))
        self.train_meta = self.dataset.extract_meta(self.table_name, self.train_df)
        self.test_meta = self.dataset.extract_meta(self.table_name, self.test_df)

    def set_args(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.use_GPU else 'cpu')

    def align(self, train_G, test_G, train_idx=None, test_idx=None):
        self.train_data = self.dataset.extract_data(self.table_name, self.train_df)
        self.test_data = self.dataset.extract_data(self.table_name, self.test_df)
        if train_idx is not None:
            self.train_data = self.train_data[train_idx]
        if test_idx is not None:
            self.test_data = self.test_data[test_idx]
        self.train_data = self.train_data[[len(row) > 0 for row in train_G]]
        self.train_G = [row for row in train_G if len(row) > 0]
        self.test_data = self.test_data[[len(row) > 0 for row in test_G]]
        self.test_G = [row for row in test_G if len(row) > 0]
        self.G = np.zeros(len(self.train_G))
        for i in range(len(self.train_G)):
            self.G[i] = len(self.train_G[i])
        logger.info(f'Begin transforming {self.table_name}')
        self.transform_train_data()
        self.transform_test_data()
        return self.train_G, self.test_G

    def transform_train_data(self):
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(self.train_data)
        self.train_data = imputer.transform(self.train_data)
        scaler = StandardScaler()
        scaler.fit(self.train_data)
        self.train_data = scaler.transform(self.train_data)
        self.imputer = imputer
        self.scaler = scaler

    def transform_test_data(self):
        self.test_data = self.imputer.transform(self.test_data)
        self.test_data = self.scaler.transform(self.test_data)

    @abc.abstractmethod
    def get_privacy_budget(self):
        pass