import torch

from .cuhk_sysu import CUHK_SYSU
from .prw import PRW

from ..utils.transforms import get_transform
from ..utils.group_by_aspect_ratio import create_aspect_ratio_groups,\
    GroupedBatchSampler


collate_fn = lambda x: x


def get_dataset(args, train=True):
    paths = {
        'CUHK-SYSU': ('data/CUHK-SYSU/', CUHK_SYSU),
        'PRW': ('data/PRW', PRW)
    }
    p, ds_cls = paths[args.dataset]

    if train:
        train_set = ds_cls(p, get_transform(args.train.use_flipped),
                           mode='train')
        return train_set
    else:
        test_set = ds_cls(p, get_transform(False),
                          mode='test')
        probe_set = ds_cls(p, get_transform(False),
                           mode='probe')
        return test_set, probe_set


def get_data_loader(args, train=True):
    dataset = get_dataset(args, train)
    if train:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)

        if args.train.aspect_grouping >= 0:
            group_ids = create_aspect_ratio_groups(
                dataset, k=args.train.aspect_grouping)
            train_batch_sampler = GroupedBatchSampler(
                train_sampler, group_ids, args.train.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.train.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers,
            collate_fn=collate_fn)
        return data_loader

    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset[0])
        probe_sampler = torch.utils.data.SequentialSampler(dataset[1])

        data_loader_test = torch.utils.data.DataLoader(
            dataset[0], batch_size=args.test.batch_size,
            sampler=test_sampler, num_workers=args.num_workers,
            collate_fn=collate_fn)
        data_loader_probe = torch.utils.data.DataLoader(
            dataset[1], batch_size=args.test.batch_size,
            sampler=probe_sampler, num_workers=args.num_workers,
            collate_fn=collate_fn)
        return data_loader_test, data_loader_probe
