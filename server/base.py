import torch
from loguru import logger
import abc
import time
import wandb


class Server:
    def __init__(self, table_names, workers):
        self.table_names = table_names
        self.workers = workers
        self.N = len(self.table_names)

    def load_data(self, dataset, data_path):
        self.dataset = dataset
        self.task = self.dataset.task
        self.C = self.dataset.num_class
        for worker in self.workers:
            worker.load_data(self.dataset, data_path)

    def set_args(self, args):
        self.args = args
        self.device = torch.device('cuda' if args.use_GPU else 'cpu')
        logger.info('Using device: {}'.format(self.device))
        for worker in self.workers:
            worker.set_args(args)

    def build_mapping(self):
        # Step 1: Table mapping
        self.f, self.original_G, self.b = self._build_mapping('train')
        self.test_f, self.original_test_G, self.test_b = self._build_mapping('test')
        self.M = self.b.shape[0]
        self.test_M = self.test_b.shape[0]
        self.aligned_G = []
        self.aligned_test_G = []
        for _ in range(self.N):
            self.aligned_G.append([[j] for j in range(0, self.M)])
            self.aligned_test_G.append([[j] for j in range(0, self.test_M)])

    def _build_mapping(self, type):
        logger.info(f'Begin building {type} mapping')
        table_meta = {}
        label = None
        for i in range(self.N):
            table_name = self.workers[i].table_name
            if type == 'train':
                table_meta[table_name], b = self.workers[i].train_meta
                if b is not None: label = b
            elif type == 'test':
                table_meta[table_name], b = self.workers[i].test_meta
                if b is not None: label = b
        f, G, table_index_mapping = self.dataset.build_mapping(table_meta)
        label = label[table_index_mapping[list(self.dataset.label_info.keys())[0]]]
        return f, G, label

    def train(self, epoch):
        self.init()
        avg_exec_t = 0.
        if self.task == 'classification':
            max_train_acc = 0.
            max_test_acc = 0.
        elif self.task == 'regression':
            min_train_rmse = 1e9
            min_test_rmse = 1e9
        if self.args.use_DP:
            final_privacy_budget = 0.
        logger.info('Start iteration')
        for epoch in range(1, epoch + 1):
            start = time.time()
            self.train_per_epoch()
            # Validate
            exec_t = time.time() - start
            avg_exec_t += exec_t
            item = {
                'epoch': epoch,
                'exec_t': exec_t,
            }
            if self.args.use_DP:
                privacy_budget = 0
                for worker in self.workers:
                    privacy_budget = max(privacy_budget, worker.get_privacy_budget())
                item['privacy_budget'] = privacy_budget
                final_privacy_budget = privacy_budget   
            if self.task == 'classification':
                train_loss, train_acc = self.validate()
                test_loss, test_acc = self.test()
                max_train_acc = max(max_train_acc, train_acc)
                max_test_acc = max(max_test_acc, test_acc)
                item.update({
                    'train/loss': train_loss,
                    'train/acc': train_acc,
                    'test/loss': test_loss,
                    'test/acc': test_acc
                })
                if self.args.use_DP:
                    logger.info(
                        '[Epoch %2d] train_loss = %.3f, train_acc = %.3f, test_loss = %.3f, test_acc = %.3f, epsilon = %.3f, exec_t = %.2fs'
                        % (
                            item['epoch'],
                            item['train/loss'],
                            item['train/acc'],
                            item['test/loss'],
                            item['test/acc'],
                            item['privacy_budget'],
                            round(item['exec_t'], 2),
                        )
                    )
                else:
                    logger.info(
                        '[Epoch %2d] train_loss = %.3f, train_acc = %.3f, test_loss = %.3f, test_acc = %.3f, exec_t = %.2fs'
                        % (
                            item['epoch'],
                            item['train/loss'],
                            item['train/acc'],
                            item['test/loss'],
                            item['test/acc'],
                            round(item['exec_t'], 2),
                        )
                    )
            elif self.task == 'regression':
                train_rmse = self.validate()
                test_rmse = self.test()
                min_train_rmse = min(min_train_rmse, train_rmse)
                min_test_rmse = min(min_test_rmse, test_rmse)
                item.update({
                    'train/rmse': train_rmse,
                    'test/rmse': test_rmse,
                })
                if self.args.use_DP:
                    logger.info(
                        '[Epoch %2d] train_rmse = %.3f, test_rmse = %.3f, epsilon = %.3f, exec_t = %.2fs'
                        % (
                            item['epoch'],
                            item['train/rmse'],
                            item['test/rmse'],
                            item['privacy_budget'],
                            round(item['exec_t'], 2),
                        )
                    )
                else:
                    logger.info(
                        '[Epoch %2d] train_rmse = %.3f, test_rmse = %.3f, exec_t = %.2fs'
                        % (
                            item['epoch'],
                            item['train/rmse'],
                            item['test/rmse'],
                            round(item['exec_t'], 2),
                        )
                    )
            wandb.log(item, step=epoch)
            if self.task == 'classification':
                if train_loss > 100:
                    break
            elif self.task == 'regression':
                if train_rmse > 100:
                    break
        avg_exec_t /= epoch
        logger.info('[Finish] average exec_t = %.2fs' % avg_exec_t)
        if self.task == 'classification':
            logger.info('[MaxAcc] max_train_acc = %.3f, max_test_acc = %.3f' % (max_train_acc, max_test_acc))
        elif self.task == 'regression':
            logger.info('[MinRMSE] min_train_rmse = %.3f, min_test_rmse = %.3f' % (min_train_rmse, min_test_rmse))
        if self.args.use_DP:
            logger.info('[Privacy Budget] epsilon is %.3f' % final_privacy_budget)

    def init(self):
        pass

    @abc.abstractmethod
    def train_per_epoch(self):
        pass

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass