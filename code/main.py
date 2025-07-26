from models import IEGMCL
import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd
from datetime import datetime

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="IEGMCL for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='bindingdb')
parser.add_argument('--split', default='E4', type=str, metavar='S', help="split task", choices=['E1', 'E2', 'E3', 'E4'])
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    cfg = get_cfg_defaults()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}/{cfg.SOLVER.SEED}')

    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    dataFolder = f'../datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    """load data"""
    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    print(f"train_set: {len(df_train)}")
    print(f"val_set: {len(df_val)}")
    print(f"test_set: {len(df_test)}")

    set_seed(cfg.SOLVER.SEED)
    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
                                                                   'drop_last': True, 'collate_fn': graph_collate_func}
    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = IEGMCL(**cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True


    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, args.data, args.split, **cfg)

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))


    print(f"\nDirectory for saving result: {cfg.RESULT.OUTPUT_DIR}{args.data}")
    print(f'\nend...')

    return result

if __name__ == '__main__':
    print(f"start: {datetime.now()}")
    start_time = time.time()
    """ train """
    result = main()
    """"""
    end_time = time.time()
    total_time_seconds = end_time - start_time
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60
    seconds = total_time_seconds % 60
    print("Total running time of the model: {} hours {} minutes {} seconds".format(int(hours), int(minutes),
                                                                              int(seconds)))
    print(f"end: {datetime.now()}")