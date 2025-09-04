import os
from abc import ABCMeta, abstractmethod

import torch


def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))


class AbstractBaseLogger(metaclass=ABCMeta):
    @abstractmethod
    def log(self, *args, **kwargs):
        raise NotImplementedError

    def complete(self, *args, **kwargs):
        pass


class LoggerService:
    def __init__(self, train_loggers, val_loggers):
        self.train_loggers = train_loggers
        self.val_loggers = val_loggers

    def log_train(self, log_data):
        for logger in self.train_loggers:
            logger.log(**log_data)

    def log_val(self, log_data):
        for logger in self.val_loggers:
            logger.log(**log_data)

    def complete(self, log_data):
        for logger in self.train_loggers:
            if isinstance(logger, RecentModelLogger) or isinstance(logger, BestModelLogger) or isinstance(logger, EpochModelLogger):
                logger.log(**log_data)
        for logger in self.val_loggers:
            if isinstance(logger, RecentModelLogger) or isinstance(logger, BestModelLogger) or isinstance(logger, EpochModelLogger):
                logger.log(**log_data)


class EpochModelLogger(AbstractBaseLogger):
    def __init__(self, model_checkpoint_root):
        self.model_checkpoint_root = model_checkpoint_root

    def log(self, *args, **kwargs):
        state_dict = kwargs['state_dict']
        epoch = kwargs['epoch']
        torch.save(state_dict, os.path.join(self.model_checkpoint_root, f'model_epoch_{epoch}.pth'))


class RecentModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, filename='checkpoint-recent.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.recent_epoch = None
        self.filename = filename

    def log(self, *args, **kwargs):
        epoch = kwargs['epoch']

        if self.recent_epoch != epoch:
            self.recent_epoch = epoch
            state_dict = kwargs['state_dict']
            state_dict['epoch'] = kwargs['epoch']
            save_state_dict(state_dict, self.checkpoint_path, self.filename)

    def complete(self, *args, **kwargs):
        save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename + '.final')


class BestModelLogger(AbstractBaseLogger):
    def __init__(self, checkpoint_path, metric_key='mean_iou', filename='best_acc_model.pth'):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.best_metric = 0.
        self.metric_key = metric_key
        self.filename = filename

    def log(self, *args, **kwargs):
        if self.metric_key not in kwargs:
            pass
        else:
            current_metric = kwargs[self.metric_key]
            if self.best_metric < current_metric:
                print("Update Best {} Model at {}".format(self.metric_key, kwargs['epoch']))
                self.best_metric = current_metric
                save_state_dict(kwargs['state_dict'], self.checkpoint_path, self.filename)


class MetricGraphPrinter(AbstractBaseLogger):
    def __init__(self, writer, key='train_loss', graph_name='Train Loss', group_name='metric'):
        self.key = key
        self.graph_label = graph_name
        self.group_name = group_name
        self.writer = writer

    def log(self, *args, **kwargs):
        # Removed debug print and simplified logic based on the fix in LoggerService
        if self.key in kwargs and 'accum_iter' in kwargs:
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, kwargs[self.key], kwargs['accum_iter'])
        elif 'accum_iter' in kwargs: # If key is not in kwargs, but accum_iter is, log 0
            self.writer.add_scalar(self.group_name + '/' + self.graph_label, 0, kwargs['accum_iter'])
        # No else needed, if accum_iter is also missing, we just don't log for this MetricGraphPrinter

    def complete(self, *args, **kwargs):
        self.writer.close()