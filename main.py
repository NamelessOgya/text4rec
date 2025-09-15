import torch
import os
import shutil

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
from datasets import dataset_factory


def train():
    if args.recreate_data:
        print("Recreating data...")
        
        # Raw data folder for the dataset
        raw_data_folder = os.path.join('Data', args.dataset_code)
        if os.path.exists(raw_data_folder):
            shutil.rmtree(raw_data_folder)
            print(f"Removed {raw_data_folder}")

        # Preprocessed data folder
        dataset_for_path = dataset_factory(args)
        preprocessed_folder = dataset_for_path._get_preprocessed_folder_path()
        if os.path.exists(preprocessed_folder):
            shutil.rmtree(preprocessed_folder)
            print(f"Removed {preprocessed_folder}")
        
        print("Data recreation complete.")

    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


    if args.do_test:
        trainer.test()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
