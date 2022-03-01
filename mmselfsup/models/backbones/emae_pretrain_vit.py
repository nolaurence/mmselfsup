from .mae_pretrain_vit import MAEViT
import torch
from ..builder import BACKBONES


@BACKBONES.register_module()
class EMAEViT(MAEViT):

    def forward(self, x):
        x = torch.cat(x)
        B = x.shape[0]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        x_1 = x[:B // 2]
        x_2 = x[B // 2:]

        # masking: length -> length * mask_ratio
        x_1, mask_1, ids_restore_1 = self.random_masking(x_1, self.mask_ratio)
        x_2, mask_2, ids_restore_2 = self.random_masking(x_2, self.mask_ratio)

        x = torch.cat([x_1, x_2])
        mask = torch.cat([mask_1, mask_2])
        ids_restore = torch.cat([ids_restore_1, ids_restore_2])

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.drop_after_pos(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

        return (x, mask, ids_restore)
