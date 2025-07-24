from typing import Type, Mapping, Callable, Optional, Union, Tuple, List

import torch
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torchmetrics import Metric

from tsl.predictors import Predictor
import torch.nn.functional as F

epsilon = 1e-8  # 防除零

def _prepare_mask(val, mask=None, *, mask_nans=False, mask_inf=False):
    """
    返回与 val 形状一致的 0/1 掩码张量。
    """
    if mask is None:
        mask = torch.ones_like(val, dtype=torch.bool)
    else:
        if mask.shape != val.shape:
            raise ValueError(f"mask shape {mask.shape} ≠ val shape {val.shape}")
        mask = mask.bool()

    if mask_nans:
        mask &= ~torch.isnan(val)
    if mask_inf:
        mask &= ~torch.isinf(val)
    return mask


def _masked_reduce(metric_tensor, mask):
    """
    将逐元 metric_tensor 在掩码下求和，并返回 (加权和, 有效元素个数)。
    """
    metric_tensor = torch.where(mask, metric_tensor, torch.tensor(0., device=metric_tensor.device))
    return metric_tensor.sum(), mask.sum()


def masked_mae(pred, true, mask=None, *, mask_nans=False, mask_inf=False, at=None, reduction='mean'):
    """Mean Absolute Error = ∑|pred-true| / N"""
    if at is not None:
        pred, true = pred[:, at], true[:, at]
        mask = mask[:, at] if mask is not None else None

    val = F.l1_loss(pred, true, reduction='none')  # 逐元 |Δ|
    mask = _prepare_mask(val, mask, mask_nans=mask_nans, mask_inf=mask_inf)

    if reduction == 'none':
        # 3) 把无效位置设为 0
        val = torch.where(mask, val, torch.tensor(0., device=val.device))
        return val  # -> 返回 [B, C, N, T] 或 [B, C, N, T, E] 的张量
    num, den = _masked_reduce(val, mask)
    return num / den if den > 0 else num  # 无有效元素时返回 0


class STMoEImputer(Predictor):
    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = False,
                 warmup_epoch: int = 1,
                 whiten_prob: Union[float, List[float]] = 0.2,
                 prediction_loss_weight: float = 1.0,
                 use_uncertainty: bool = True,
                 use_quantile: bool = True,
                 quantile: float = 0.7,
                 impute_only_missing: bool = True,
                 warm_up_steps: Union[int, Tuple[int, int]] = 0,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional[Type] = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(STMoEImputer, self).__init__(model_class=model_class,
                                      model_kwargs=model_kwargs,
                                      optim_class=optim_class,
                                      optim_kwargs=optim_kwargs,
                                      loss_fn=loss_fn,
                                      metrics=metrics,
                                      scheduler_class=scheduler_class,
                                      scheduler_kwargs=scheduler_kwargs)
        self.scale_target = scale_target
        self.prediction_loss_weight = prediction_loss_weight
        self.impute_only_missing = impute_only_missing

        self.use_uncertainty = use_uncertainty
        self.threshold = 0.
        self.use_quantile = use_quantile
        self.quantile = quantile
        self.warmup_epoch = warmup_epoch

        if isinstance(whiten_prob, (list, tuple)):
            self.whiten_prob = torch.Tensor(whiten_prob)
        else:
            self.whiten_prob = whiten_prob

        if isinstance(warm_up_steps, int):
            self.warm_up_steps = (warm_up_steps, 0)
        elif isinstance(warm_up_steps, (list, tuple)):
            self.warm_up_steps = tuple(warm_up_steps)
        assert len(self.warm_up_steps) == 2

    def trim_warm_up(self, *args):
        """Trim all tensors in :obj:`args` removing a number of first and last
        steps equals to :obj:`(self.warm_up_steps[0], self.warm_up_steps[1])`,
        respectively."""
        left, right = self.warm_up_steps
        trim = lambda s: s[:, left:s.size(1) - right]
        args = recursive_apply(args, trim)
        if len(args) == 1:
            return args[0]
        return args

    # Imputation data hooks ###################################################

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Rearrange batch for imputation:
            1. Move :obj:`eval_mask` from :obj:`batch.input` to :obj:`batch`
            2. Move :obj:`mask` from :obj:`batch` to :obj:`batch.input`
        """
        # move eval_mask from batch.input to batch
        batch.eval_mask = batch.input.pop('eval_mask')
        # move mask from batch to batch.input
        batch.input.mask = batch.pop('mask')
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask
        return batch

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        r"""For every training batch, randomly mask out value with probability
        :math:`p = \texttt{self.whiten\_prob}`. Then, whiten missing values in
         :obj:`batch.input.x`"""
        super(STMoEImputer, self).on_train_batch_start(batch, batch_idx)
        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)
        whiten_mask = torch.rand(mask.size(), device=mask.device) > p
        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask

    def _unpack_batch(self, batch):
        transform = batch.get('transform')
        return batch.input, batch.target, batch.eval_mask, transform

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y = batch.y
        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)
        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]
        if self.impute_only_missing:
            y_hat = torch.where(batch.mask.bool(), y, y_hat)
        output = dict(y=batch.y, y_hat=y_hat, mask=batch.eval_mask)
        return output

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = y_hat[0]
        else:
            imputation, predictions = y_hat_loss, []

        loss = self.loss_fn(imputation, y_loss, mask)
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y_loss, mask)
            loss += self.prediction_loss_weight * pred_loss

        return y_hat.detach(), y, loss

    def get_quantile_label(self, gated_loss, gate, real):
        gated_loss = gated_loss.unsqueeze(dim = -1) # B C N T 1

        real = real.permute(0, 3, 2, 1)  # -> [B, C, N, T]
        real = real.unsqueeze(-1)  # -> [B, C, N, T, 1]

        max_quantile = gated_loss.quantile(self.quantile)
        min_quantile = gated_loss.quantile(1 - self.quantile)
        incorrect = (gated_loss > max_quantile).expand_as(gate)
        correct = ((gated_loss < min_quantile) & (real > self.threshold)).expand_as(gate)
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        not_chosen = gate.topk(dim = -1, k = 2, largest = False).indices
        selected = torch.zeros_like(gate).scatter_(-1, cur_expert, 1.0)
        scaling = torch.zeros_like(gate).scatter_(-1, not_chosen, 0.5)
        selected[incorrect] = scaling[incorrect]
        l_worst_avoidance = selected.detach()
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0) * correct
        l_best_choice = selected.detach()
        return l_worst_avoidance, l_best_choice

    def get_label(self, ind_loss, gate, real):
        n_experts = gate.size(-1)
        empty_val = (real.permute(0,2,3,1).unsqueeze(-1).expand_as(gate)) <= self.threshold
        max_error = ind_loss.argmax(dim = -1, keepdim = True)
        cur_expert = gate.argmax(dim = -1, keepdim = True)
        incorrect = max_error == cur_expert
        selected = torch.zeros_like(gate).scatter(-1, cur_expert, 1.0)
        scaling = torch.ones_like(gate) * ind_loss
        scaling = scaling.scatter(-1, max_error, 0.)
        scaling = scaling / (scaling.sum(dim = -1, keepdim = True)) * (1 - selected)
        l_worst_avoidance = torch.where(incorrect, scaling, selected)
        l_worst_avoidance = torch.where(empty_val, torch.zeros_like(gate), l_worst_avoidance)
        l_worst_avoidance = l_worst_avoidance.detach()
        min_error = ind_loss.argmin(dim = -1, keepdim = True)
        correct = min_error == cur_expert
        scaling = torch.zeros_like(gate)
        scaling = scaling.scatter(-1, min_error, 1.)
        l_best_choice = torch.where(correct, selected, scaling)
        l_best_choice = torch.where(empty_val, torch.zeros_like(gate), l_best_choice)
        l_best_choice = l_best_choice.detach()
        return l_worst_avoidance, l_best_choice

    def get_uncertainty(self, x, mode = 'psd', threshold = 0.0):

        def _acorr(x, dim = -1):
            size = x.size(dim)
            x_fft = torch.fft.fft(x, dim = dim)
            acorr = torch.fft.ifft(x_fft * x_fft.conj(), dim = dim).real
            return acorr / (size ** 2)

        def nanstd(x, dim, keepdim = False):
            return torch.sqrt(
                        torch.nanmean(
                                torch.pow(torch.abs(x - torch.nanmean(x, dim = dim, keepdim = True)), 2),
                                dim = dim, keepdim = keepdim
                            )
                    )

        with torch.no_grad():
            if mode == 'acorr':
                std = x.std(dim = -2, keepdim = True)
                corr = _acorr(x, dim = -2)
                x_noise = x + std * torch.randn((1,1,x.size(-2),1), device = x.device) / 2
                corr_w_noise = _acorr(x_noise, dim = -2)
                corr_changed = torch.abs(corr - corr_w_noise)
                uncertainty = torch.ones_like(corr_changed) * (corr_changed > corr_changed.quantile(1 - self.quantile))
            elif mode == 'psd':
                from copy import deepcopy as cp
                vals = cp(x)
                vals[vals <= threshold] = torch.nan
                diff = vals[:,:,1:] - vals[:,:,:-1]
                corr_changed = torch.nanmean(torch.abs(diff), dim = -2, keepdim = True) / (nanstd(diff, dim = -2, keepdim = True) + 1e-6)
                corr_changed[corr_changed != corr_changed] = 0.
                uncertainty = torch.ones_like(corr_changed) * (corr_changed < corr_changed.quantile(self.quantile))
            else:
                raise NotImplementedError
            return uncertainty
        return None


    def training_step(self, batch, batch_idx):
        cur_epoch = self.current_epoch
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, batch.original_mask)

        imputation, gate, (imps, preds) = y_hat_loss

        #  形状对齐：gate 扩到与 imps 同维 [B,C,N,T,E]
        gate = gate.permute(0, 3, 1, 2, 4)  # (B,N,T,C,E) -> (B,C,N,T,E)
        # ---- 计算主插补损失 ----
        loss_imp = masked_mae(imputation, y_loss, mask, reduction='none')

        # ---- 计算专家损失 ----
        E = imps.size(-1)
        loss_expert_list = []

        for e in range(E):
            imp_e, pred_e = imps[..., e], preds[..., e]
            loss = masked_mae(imp_e, y_loss, mask, reduction='none')
            mae_e = masked_mae(imp_e,y,batch.eval_mask, reduction='none')
            self.log(f'train_expert_{e}_mae', mae_e.mean(), on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=True)
            # 由 [B, 4, T, N, C] -> [4, B, T, N, C]
            pred_e = pred_e.permute(1, 0, 2, 3, 4)
            for pred in pred_e:
                loss += self.prediction_loss_weight * masked_mae(pred, y_loss, mask, reduction='none')
            loss_expert_list.append(loss)

        expert_losses = torch.stack(loss_expert_list, dim=0)
        baseline_loss = expert_losses.mean()

        # 4) 生成 “最差 / 最佳” 标签（沿 E 维）
        if self.use_quantile:
            # loss_imp 原先是 [B, T, N, C]->[B, C, N, T]
            gated_loss = loss_imp.permute(0, 3, 2, 1)
            l_worst_avoidance, l_best_choice = self.get_quantile_label(gated_loss, gate, y_loss)
        # else:
        #     l_worst_avoidance, l_best_choice = self.get_label(ind_loss, gate, target)

        if self.use_uncertainty:
            # target B T N C - > B C N T
            uncertainty = self.get_uncertainty(y_loss.permute(0,3,2,1), threshold = 0)
            uncertainty = uncertainty.unsqueeze(dim = -1)
        else:
            uncertainty = torch.ones_like(gate)
        for e, l in enumerate(loss_expert_list):
            self.log(f'train_expert_{e}_loss', l.mean(), on_step=False, on_epoch=True, prog_bar=False, rank_zero_only=True)

        # Gate 交叉熵惩罚 / 奖励
        worst_avoidance = -.5 * l_worst_avoidance * torch.log(gate) * (2 - uncertainty)
        best_choice = -.5 * l_best_choice * torch.log(gate) * uncertainty

        if cur_epoch <= self.warmup_epoch:
            loss = baseline_loss.mean()
        else:
            loss = baseline_loss.mean() + worst_avoidance.mean() + best_choice.mean()

        # epoch 1 logging seems not useful
        # Logging
        self.train_metrics.update(imputation, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss


    def validation_step(self, batch, batch_idx):

        y_hat, y, val_loss = self.shared_step(batch, batch.mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y, training_mask, eval_mask = batch.y, batch.mask, batch.eval_mask
        test_loss = self.loss_fn(y_hat, y, training_mask)  # reconstruction loss

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss


    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.2)
        parser.add_argument('--prediction-loss-weight', type=float, default=1.0)
        parser.add_argument('--impute-only-missing', type=bool, default=True)
        parser.add_argument('--warm-up-steps', type=tuple, default=(0, 0))
        return parser
