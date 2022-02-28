# model settings
model = dict(
    type='EMAE',
    backbone=dict(type='EMAEViT', arch='b', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type='EMAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(
        type='EMAEPretrainHead',
        norm_pix=True,
        patch_size=16,
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        temperature=0.2,
        lamb=1.0))
