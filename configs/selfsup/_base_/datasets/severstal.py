# dataset settings
data_source = 'Severstal'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.2, 1.0), interpolation=3),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    imgs_per_gpu=64,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='severstal/train',
            ann_file='severstal/train_out.csv',
        ),
        pipeline=train_pipeline,
        prefetch=prefetch))
