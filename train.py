import datetime
import os

from catalyst import dl
from catalyst.contrib.nn.schedulers import OneCycleLRWithWarmup
from torch import nn, optim
from torch.utils.data import DataLoader

from data import NpOcrDataset, LabelEncoder, get_train_transform, get_valid_transform
from evaluate import evaluate_model
from model import MultiheadClassifier
from runner import MultiheadClassificationRunner, get_runner_callbacks
from utils import seed_everything, seed_worker

if __name__ == "__main__":
    seed_everything()
    
    class CFG:
        dataset_folder = "autoriaNumberplateOcrRu-2020-10-12"
        processing_size = (300, 65)
        lp_maxlen = 9
        backbone = "resnet18"
        num_epochs = 10
        batch_size = 128
        lr_min = 1e-4
        lr_max = 1e-2
        num_workers = 4
        logdir=f"run_{datetime.datetime.now()}"
    
    label_encoder = LabelEncoder(max_len=CFG.lp_maxlen)
    train_dataset = NpOcrDataset(
        os.path.join(CFG.dataset_folder, "train"),
        get_train_transform(CFG.processing_size),
        label_encoder
    )
    valid_dataset = NpOcrDataset(
        os.path.join(CFG.dataset_folder, "val"),
        get_valid_transform(CFG.processing_size),
        label_encoder
    )
    test_dataset = NpOcrDataset(
        os.path.join(CFG.dataset_folder, "test"),
        get_valid_transform(CFG.processing_size),
        label_encoder
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    loaders = { "train": train_loader, "valid": valid_loader }

    model = MultiheadClassifier(
        backbone_name=CFG.backbone,
        backbone_pretrained=False,
        input_size=CFG.processing_size,
        num_heads=CFG.lp_maxlen,
        num_classes=len(label_encoder.alphabet)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr_min)
    scheduler = OneCycleLRWithWarmup(
        optimizer=optimizer,
        num_steps=CFG.num_epochs,
        lr_range=(CFG.lr_max, CFG.lr_min),
        init_lr=CFG.lr_min
    )

    runner = MultiheadClassificationRunner(num_heads=CFG.lp_maxlen)
    callbacks = get_runner_callbacks(
        num_heads=CFG.lp_maxlen,
        num_classes_per_head=len(label_encoder.alphabet),
        class_names=label_encoder.alphabet,
        logdir=CFG.logdir
    )

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=CFG.num_epochs,
        verbose=True,
        callbacks=callbacks,
        loggers={ "console": dl.ConsoleLogger(), "tb": dl.TensorboardLogger(CFG.logdir) },
        load_best_on_end=True
    )

    evaluate_model(runner.model, test_dataset, label_encoder)
