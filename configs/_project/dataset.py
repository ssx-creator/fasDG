
dataset_type_2 = 'VOCDataset'

data_root_test_t1 = 'xxx/Daytime-Foggy/VOC2007/'
data_root_test_t2 = 'xxx/Dusk-rainy/VOC2007/'
data_root_test_t3 = 'xxxx/Night_rainy/VOC2007/'
data_root_test_t4 = 'xxxxx/Night-Sunny/VOC2007/'


img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1067, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type="ImageToTensor", keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=[],
    val=[ ],
    test=[

        # dict(
        #     type=dataset_type_2,
        #     samples_per_gpu=2,
        #     ann_file=data_root_test_t1 + 'ImageSets/Main/train.txt',
        #     img_prefix=data_root_test_t1,
        #     pipeline=test_pipeline),

        # dict(
        #     type=dataset_type_2,
        #     samples_per_gpu=2,
        #     ann_file=data_root_test_t2 + 'ImageSets/Main/train.txt',
        #     img_prefix=data_root_test_t2,
        #     pipeline=test_pipeline),

        # dict(
        #     type=dataset_type_2,
        #     samples_per_gpu=2,
        #     ann_file=data_root_test_t3 + 'ImageSets/Main/train.txt',
        #     img_prefix=data_root_test_t3,
        #     pipeline=test_pipeline),
        
        dict(
            type=dataset_type_2,
            samples_per_gpu=2,
            ann_file=data_root_test_t4 + 'ImageSets/Main/train.txt',
            img_prefix=data_root_test_t4,
            pipeline=test_pipeline
            ),
        ]

        )
            