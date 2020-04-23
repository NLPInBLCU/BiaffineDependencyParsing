import bisect
import warnings
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import IterableDataset
import numpy as np
from typing import List


class ConcatTensorRandomDataset(Dataset):
    r"""
    实现了对不同源的TensorData的指数平滑采样，
    注意：
    （1）仅仅可以在训练模式下使用
    （2）无法保证同样的idx取出同样的数据
    （3）数据取样过程是一个随机过程，无论DataLoader中使用什么样的Sampler都不影响
    （4）采样时，首先决定从哪个源取样（根据权值或者指数平滑概率），确定目标源之后，会随机从目标源采样
    """

    def __init__(self, datasets: List[TensorDataset], probs: List[float] = None, exp: float = None, mode: str = 'exp'):
        """

        :param datasets: 各个源本身的Data Set
        :param probs: 按照概率采样，对应每个源的概率，长度等于datasets的数量
        :param exp: 按照指数平滑采样，0<exp<1
        :param mode:指示是采用概率采样还是采用指数平滑采样
        """
        super().__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        assert mode in ['prob', 'exp'], 'ConcatTensorRandomDataset mode只能为prob或者exp'
        if mode == 'prob':
            assert probs and len(probs) == len(datasets) and sum(probs) == 1
        else:
            assert exp and 0 < exp < 1
        self.datasets = list(datasets)
        self.dataset_idxs = list(range(len(self.datasets)))
        self.dataset_lens = [len(x) for x in self.datasets]
        self.original_lengths = []  # 记录每个源的原始数据长度
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
            self.original_lengths.append(len(d))
        if mode == 'exp':
            original_probs = self.original_lengths / np.sum(self.original_lengths)
            # 指数加权
            probs_exp = original_probs ** exp
            # softmax
            pes = np.exp(probs_exp)
            self.probs = pes / np.sum(pes)
        else:
            assert isinstance(probs, list) and probs
            self.probs = np.array(probs)
        self.sample_total_length = np.sum(self.original_lengths * self.probs)

    def __len__(self):
        return int(self.sample_total_length)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        # 依据概率决定从哪个源采样
        target_dataset_idx = np.random.choice(self.dataset_idxs, p=self.probs)
        # 从目标源采样
        sample_idx = idx % self.dataset_lens[target_dataset_idx]
        return self.datasets[target_dataset_idx][sample_idx]
