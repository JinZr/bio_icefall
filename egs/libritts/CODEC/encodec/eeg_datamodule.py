import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    SpeechSynthesisDataset,
)
from lhotse.dataset.input_strategies import AudioSamples
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from icefall.utils import str2bool

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class EEGDataModule:
    """
    EEG 数据加载模块，适用于自监督音频任务（无转录文本）。
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # 使用 AudioSamples 作为输入策略
        self.args.input_strategy = "AudioSamples"
        self.args.num_buckets = 5

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="音频数据相关选项",
            description="这些选项用于 Lhotse CutSet 的 PyTorch DataLoader 的准备，控制批量大小、采样策略、数据增强等。",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("/nvme4/lsz/mix_v1_audio/test"),
            help="包含音频 manifest 的目录路径。",
        )
        group.add_argument(
            "--max-duration",
            type=float,
            default=200.0,
            help="单个批次中池化的最大录音时长（秒）。",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
            help="动态采样器中的桶的数量。",
        )
        group.add_argument(
            "--shuffle",
            type=bool,
            default=True,
            help="是否在每个 epoch 中打乱数据。",
        )
        group.add_argument(
            "--drop-last",
            type=bool,
            default=True,
            help="是否丢弃最后一个批次。",
        )
        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
            help="DataLoader 的工作线程数。",
        )
        group.add_argument(
            "--full-libri",
            type=str2bool,
            default=True,
            help="""When enabled, use the entire LibriTTS training set. 
            Otherwise, use the clean-100 subset.""",
        )
    def train_dataloaders(self, cuts_train: CutSet, world_size: Optional[int] = None, rank: Optional[int] = None) -> DataLoader:
        """
        创建训练集的 DataLoader。
        """
        logging.info("正在创建训练数据集")
        train_dataset = SpeechSynthesisDataset(
            feature_input_strategy=AudioSamples(),  # 使用音频样本作为输入策略
            return_cuts=True,
            return_text=False,
        )

        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=self.args.shuffle,
            num_buckets=self.args.num_buckets,
            drop_last=self.args.drop_last,
            world_size=world_size,
            rank=rank,
        )

        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        return DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            worker_init_fn=worker_init_fn,
        )

    def valid_dataloaders(
        self, 
        cuts_valid: CutSet, 
        world_size: Optional[int] = None, 
        rank: Optional[int] = None,
    ) -> DataLoader:
        """
        创建验证集的 DataLoader。
        """
        logging.info("正在创建验证数据集")
        valid_dataset = SpeechSynthesisDataset(
            feature_input_strategy=AudioSamples(),
            return_cuts=True,
            return_text=False,
        )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
            world_size=world_size,
            rank=rank,
        )

        return DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloaders(self, cuts_test: CutSet) -> DataLoader:
        """
        创建测试集的 DataLoader。
        """
        logging.info("正在创建测试数据集")
        test_dataset = SpeechSynthesisDataset(
            feature_input_strategy=AudioSamples(),
            return_cuts=True,
            return_text=False,
        )

        test_sampler = DynamicBucketingSampler(
            cuts_test,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=None,
            num_workers=1,
            drop_last=False,
            persistent_workers=True,
        )

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def train_all_shuf_cuts(self) -> CutSet:
        logging.info(
            "About to get the shuffled train-clean-100, \
            train-clean-360 and train-other-500 cuts"
        )
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest_lazy(
            self.args.manifest_dir / "cuts_manifest.jsonl.gz"
        )
