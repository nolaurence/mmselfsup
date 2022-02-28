# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import build_neck
from mmselfsup.utils import concat_all_gather
import torch.distributed as dist

from ..builder import HEADS


@HEADS.register_module()
class EMAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

    def __init__(self,
                 norm_pix=False,
                 patch_size=16,
                 predictor=None,
                 temperature=0.1,
                 lamb=1.0):
        super(EMAEPretrainHead, self).__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size

        assert predictor is not None

        self.predictor = build_neck(predictor)
        self.T = temperature
        self.lamb = lamb

    def patchify(self, imgs):

        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward(self, x, pred, mask, cls_token):
        losses = dict()

        # Mask image modeling loss
        target = self.patchify(x)
        if self.norm_pix:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss_mim = (pred - target)**2
        loss_mim = loss_mim.mean(dim=-1)

        loss_mim = (loss_mim * mask).sum() / mask.sum()
        losses['mim_loss'] = loss_mim

        # Contrastive loss
        B = cls_token.shape[0]
        features = self.predictor(cls_token)
        features_q = features[:B // 2]
        features_k = features[B // 2:]

        features_q = nn.functional.normalize(features_q, dim=1)
        features_k = nn.functional.normalize(features_k, dim=1)

        # gather all targets
        features_k = concat_all_gather(features_k)
        logits = torch.einsum('nc,mc->nm', [features_q, features_k]) / self.T
        labels = (torch.arange(B // 2, dtype=torch.long) +
                  B // 2 * dist.get_rank()).cuda()
        losses['contrastive_loss'] = nn.CrossEntropyLoss()(logits, labels)

        # total loss
        losses['loss'] = losses[
            'mim_loss'] + self.lamb * losses['contrastive_loss']

        return losses
