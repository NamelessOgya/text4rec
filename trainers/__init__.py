from .bert import BERTTrainer
from .dae import DAETrainer
from .vae import VAETrainer
from .sasrec import SASTrainer


TRAINERS = {
    BERTTrainer.code(): BERTTrainer,
    DAETrainer.code(): DAETrainer,
    VAETrainer.code(): VAETrainer,
    SASTrainer.code(): SASTrainer,
}


def trainer_factory(args, model, train_loader, val_loader, test_loader, export_root):
    trainer = TRAINERS[args.trainer_code]
    return trainer(args, model, train_loader, val_loader, test_loader, export_root)
