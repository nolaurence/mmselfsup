# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ALGORITHMS
from .mae import MAE


@ALGORITHMS.register_module()
class EMAE(MAE):
    """EMAE."""

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred, cls_token = self.neck(latent, ids_restore)
        losses = self.head(img, pred, mask, cls_token)

        return losses
