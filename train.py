import argparse
import datetime
import os

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import ConcatDataset, DataLoader

import data
import engine
from utils.util import archive_files, load_config_with_cli, seed_everything


def build_dataloader(conf):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        for sub_set in sub_data.sub_sets:
            train_dataset = eval(sub_data.target)(
                sub_data.data_root,
                conf.datasets.train.trsf,
                subset=sub_set,
                split=sub_data.split,
            )
            train_datasets.append(train_dataset)
    train_dataset = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        for sub_set in sub_data.sub_sets:
            val_dataset = eval(sub_data.target)(
                sub_data.data_root,
                conf.datasets.val.trsf,
                subset=sub_set,
                split=sub_data.split,
            )
            val_datasets.append(val_dataset)
    val_dataset = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf.datasets.train.batch_size,
        shuffle=True,
        num_workers=conf.datasets.train.loader_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=conf.datasets.val.batch_size,
        shuffle=False,
        num_workers=conf.datasets.val.loader_workers,
    )
    return train_loader, val_loader


def build_logger(conf, run_name):
    logger_conf = conf.get("logger", {})
    if logger_conf.get("use_wandb", False):
        try:
            if logger_conf.get("offline", False):
                os.environ.setdefault("WANDB_MODE", "offline")
            return WandbLogger(
                name=run_name,
                project=logger_conf.get("project", "PoundNet"),
                save_dir=logger_conf.get("save_dir", "logs"),
            )
        except Exception as exc:
            print(f"Falling back to CSVLogger because WandB logger setup failed: {exc}")

    return CSVLogger(save_dir="logs", name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--cfg", type=str, required=True)
    args, cfg_args = parser.parse_known_args()

    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    seed_everything(conf.train.seed)
    train_loader, val_loader = build_dataloader(conf)

    run_name = conf.name + "_" + datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    logger = build_logger(conf, run_name)

    if os.getenv("LOCAL_RANK", "0") == "0":
        archive_files(run_name, exclude_dirs=["logs", "wandb", ".git", "exp_results"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_ap_epoch",
        dirpath=os.path.join("logs", run_name),
        filename="{epoch:02d}-{val_ap_epoch:.2f}",
        save_top_k=1,
        mode="max",
    )

    model = eval(conf.train.pipeline)(opt=conf)

    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=conf.train.train_epochs,
        accelerator=conf.train.get("accelerator", "gpu"),
        devices=conf.train.gpu_ids,
        accumulate_grad_batches=conf.train.get("gradient_accumulation_steps", 1),
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=conf.train.check_val_every_n_epoch,
        precision=conf.train.get("precision", "16"),
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(os.path.join("logs", run_name, "last.ckpt"))
