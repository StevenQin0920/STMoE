import copy
import datetime
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning import strategies

from imputers.imputer import STMoEImputer
from imputers.stmoe_imputer import SingleSTMoEImputer
from models.model import STMoE
from tsl import config, logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.preprocessing import StandardScaler
from tsl.imputers import Imputer
from tsl.nn.metrics import MaskedMetric, MaskedMAE, MaskedMSE, MaskedMRE
from tsl.nn.utils import casting
from tsl.utils import parser_utils, numpy_metrics
from tsl.utils.parser_utils import ArgParser

from imputation_ops import add_missing_values
from datasets import AirQuality, MetrLA, PemsBay, PeMS03, PeMS04, PeMS07, PeMS08

import warnings

from util import load_adj, transition_matrix

warnings.filterwarnings("ignore")


def get_model_classes(model_str):
    if model_str == 'STMoE':
        model, filler = STMoE, SingleSTMoEImputer
        # model, filler = STMoE, STMoEImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name: str):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    # build missing dataset
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_sparse'):
        p_fault, p_noise = 0., 0.9  # 0.6 0.7, 0.8, 0.9
        dataset_name = dataset_name[:-7]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems03':
        return add_missing_values(PeMS03(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems04':
        return add_missing_values(PeMS04(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems07':
        return add_missing_values(PeMS07(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems08':
        return add_missing_values(PeMS08(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")


# 可能要改哦
def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--precision', type=str, default="32")
    parser.add_argument("--model-name", type=str, default='STMoE')
    parser.add_argument("--dataset-name", type=str, default='pems04_point')  # _point _block _sparse air36
    parser.add_argument("--config", type=str, default='air36.yaml')  # PEMS-BAY \METR-LA \air

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)
    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, imputer_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = imputer_cls.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # script flags
    model_cls, imputer_class = get_model_classes(args.start_up['model_name'])
    dataset = get_dataset(args.dataset_name)

    logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    if args.dataset_name.startswith('air'):
        ori_data = dataset.df
        steps = 24
    elif args.dataset_name.startswith("la"):
        ori_data, _, _ = MetrLA().load()
        steps = 288
    elif args.dataset_name.startswith("bay"):
        ori_data, _, _ = PemsBay().load()
        steps = 288

    def make_x_long(df: pd.DataFrame, steps_per_day: int, n_long: int, direction: str = "prev"):
        """
        构建x_long: 对每个时刻/节点，拼出历史n天同一时刻的特征。

        Args:
            df: shape (T, N), index为时间，columns为节点
            steps_per_day: 每天多少个时间步
            n_long: 取几天历史
            direction: "prev" 表示往前取，"next"表示往后取
        Returns:
            x_long: ndarray, shape (T, N, n_long)
        """
        T, N = df.shape
        data = df.values  # shape: [T, N]
        x_long = np.zeros((T, N, n_long), dtype=data.dtype)

        for i in range(T):
            for j in range(n_long):
                if direction == "prev":
                    idx = i - (j + 1) * steps_per_day
                    # 如果向前不够就向后补
                    if idx < 0:
                        idx = i + (j + 1) * steps_per_day
                    # Clamp，防止越界
                    idx = max(0, min(idx, T - 1))
                elif direction == "next":
                    idx = i + (j + 1) * steps_per_day
                    # 如果向后不够就向前补
                    if idx >= T:
                        idx = i - (j + 1) * steps_per_day
                    idx = max(0, min(idx, T - 1))
                else:
                    raise ValueError("direction must be 'prev' or 'next'")
                x_long[i, :, j] = data[idx, :]
        return x_long

    x_long = make_x_long(ori_data, steps, 6, 'prev')

    def build_tid_dow_mats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        T, N = df.shape

        # Time-of-day 归一化 (0~1)
        tid_vec = ((df.index.to_numpy() - df.index.normalize().to_numpy())
                   / np.timedelta64(1, "D")).astype(np.float32)  # shape (T,)

        # Day-of-week 0~6
        dow_vec = df.index.dayofweek.to_numpy().astype(np.int8)  # shape (T,)

        # 扩充到 (T, N)
        tid_mat = np.tile(tid_vec[:, None], (1, N))  # 或  np.repeat(tid_vec[:, None], N, axis=1)
        dow_mat = np.tile(dow_vec[:, None], (1, N))

        return tid_mat, dow_mat

    tid_mat, dow_mat = build_tid_dow_mats(ori_data)
    time_emb = dataset.datetime_encoded(['day', 'week']).values

    exog_map = {'longterm_feature': x_long,
                'tid_feature': tid_mat,
                'dow_feature': dow_mat,
                'global_temporal_encoding': time_emb
                }

    input_map = {
        'u': 'longterm_feature',
        'tid': 'tid_feature',
        'dow': 'dow_feature',
        't_encode': 'temporal_encoding',
        'x': 'data'
    }

    # adj = dataset.get_connectivity(threshold=args.adj_threshold,
    #                                    include_self=False,
    #                                    force_symmetric=False,
    #                                    layout = 'dense')
    # args.dynamic_args['adjs'] = [torch.tensor(i,dtype=torch.float32) for i in adj]

    if args.dataset_name.startswith("la"):
        adjs, adj = load_adj(os.path.join('adj', 'la_adj.pkl'), 'doubletransition')
        np.fill_diagonal(adj, 0.)
        args.dynamic_args['adjs'] = [torch.tensor(i) for i in adjs]
    if args.dataset_name.startswith("bay"):
        adjs, adj = load_adj(os.path.join('adj', 'bay_adj.pkl'), 'doubletransition')
        np.fill_diagonal(adj, 0.)
        args.dynamic_args['adjs'] = [torch.tensor(i) for i in adjs]
    if args.dataset_name.startswith("air"):
        adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                       include_self=False,
                                       force_symmetric=False,
                                       layout='dense')
        adjs = [transition_matrix(adj).T]
        args.dynamic_args['adjs'] = [torch.tensor(i, dtype=torch.float32) for i in adj]
    # instantiate dataset
    torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      # connectivity=adj,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len,
                                    test_len=args.test_len)

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size // args.split_batch_in)
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    adj=adj,
                                    input_size=dm.n_channels,
                                    u_size=4,
                                    output_size=dm.n_channels,
                                    window_size=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class,
                                                       return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=CosineAnnealingLR,
        scheduler_kwargs=dict(eta_min=0.1 * args.lr, T_max=args.epochs),
        **imputer_kwargs
    )

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1,
                                          monitor='val_mae', mode='min')

    tb_logger = TensorBoardLogger(logdir, name="model")

    # Explicitly specify the process group backend if you choose to

    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=tb_logger,
                         precision=args.precision,
                         accumulate_grad_batches=args.split_batch_in,
                         accelerator="gpu",
                         devices="auto",
                         # strategy=strategies.DDPStrategy(find_unused_parameters=True),
                         gradient_clip_val=args.grad_clip_val,
                         limit_train_batches=args.batches_epoch * args.split_batch_in,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(imputer,
                # ckpt_path='log/la_point/STMoE/20250711T112443_3407/epoch=4-step=1500.ckpt',
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(
                    batch_size=args.batch_inference))

    ########################################
    # testing                              #
    ########################################

    # imputer.load_model(checkpoint_callback.best_model_path)
    # imputer.freeze()
    # trainer.test(imputer,
    #              ckpt_path="log/air36/STMoE/20250716T152403_3407/epoch=137-step=21666.ckpt",
    #              dataloaders=dm.test_dataloader(
    #     batch_size=args.batch_inference))

    # output = trainer.predict(imputer, dataloaders=dm.test_dataloader(
    #     batch_size=args.batch_inference))
    # output = casting.numpy(output)
    # y_hat, y_true, mask = output['y_hat'].squeeze(-1), \
    #                       output['y'].squeeze(-1), \
    #                       output['mask'].squeeze(-1)
    # check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    # print(f'Test MAE: {check_mae:.2f}')
    # return y_hat


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
