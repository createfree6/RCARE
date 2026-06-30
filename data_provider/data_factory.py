from __future__ import annotations

from torch.utils.data import DataLoader

from data_provider.data_loader import CAREDataset


def data_provider(args, flag: str):
    shuffle_flag = flag == "train"
    drop_last = flag == "train" and getattr(args, "drop_last", False)
    batch_size = args.batch_size if flag == "train" else args.eval_batch_size
    data_set = CAREDataset(args, flag)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
