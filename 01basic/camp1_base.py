policy_imagenet = [[{
    'type': 'Posterize',
    'bits': 4,
    'prob': 0.4
}, {
    'type': 'Rotate',
    'angle': 30.0,
    'prob': 0.6
}],
                   [{
                       'type': 'Solarize',
                       'thr': 113.77777777777777,
                       'prob': 0.6
                   }, {
                       'type': 'AutoContrast',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.6
                   }, {
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Solarize',
                       'thr': 142.22222222222223,
                       'prob': 0.2
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Solarize',
                       'thr': 170.66666666666666,
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Posterize',
                       'bits': 6,
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 10.0,
                       'prob': 0.2
                   }, {
                       'type': 'Solarize',
                       'thr': 28.444444444444443,
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.6
                   }, {
                       'type': 'Posterize',
                       'bits': 5,
                       'prob': 0.4
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }, {
                       'type': 'ColorTransform',
                       'magnitude': 0.0,
                       'prob': 0.4
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 30.0,
                       'prob': 0.4
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.0
                   }, {
                       'type': 'Equalize',
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Invert',
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.4,
                       'prob': 0.6
                   }, {
                       'type': 'Contrast',
                       'magnitude': 0.8,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Rotate',
                       'angle': 26.666666666666668,
                       'prob': 0.8
                   }, {
                       'type': 'ColorTransform',
                       'magnitude': 0.2,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.8,
                       'prob': 0.8
                   }, {
                       'type': 'Solarize',
                       'thr': 56.888888888888886,
                       'prob': 0.8
                   }],
                   [{
                       'type': 'Sharpness',
                       'magnitude': 0.7,
                       'prob': 0.4
                   }, {
                       'type': 'Invert',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Shear',
                       'magnitude': 0.16666666666666666,
                       'prob': 0.6,
                       'direction': 'horizontal'
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.0,
                       'prob': 0.4
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.4
                   }, {
                       'type': 'Solarize',
                       'thr': 142.22222222222223,
                       'prob': 0.2
                   }],
                   [{
                       'type': 'Solarize',
                       'thr': 113.77777777777777,
                       'prob': 0.6
                   }, {
                       'type': 'AutoContrast',
                       'prob': 0.6
                   }],
                   [{
                       'type': 'Invert',
                       'prob': 0.6
                   }, {
                       'type': 'Equalize',
                       'prob': 1.0
                   }],
                   [{
                       'type': 'ColorTransform',
                       'magnitude': 0.4,
                       'prob': 0.6
                   }, {
                       'type': 'Contrast',
                       'magnitude': 0.8,
                       'prob': 1.0
                   }],
                   [{
                       'type': 'Equalize',
                       'prob': 0.8
                   }, {
                       'type': 'Equalize',
                       'prob': 0.6
                   }]]
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=1))
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Posterize',
            'bits': 4,
            'prob': 0.4
        }, {
            'type': 'Rotate',
            'angle': 30.0,
            'prob': 0.6
        }],
                  [{
                      'type': 'Solarize',
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 170.66666666666666,
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Posterize',
                      'bits': 6,
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 10.0,
                      'prob': 0.2
                  }, {
                      'type': 'Solarize',
                      'thr': 28.444444444444443,
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.6
                  }, {
                      'type': 'Posterize',
                      'bits': 5,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 30.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.0
                  }, {
                      'type': 'Equalize',
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Rotate',
                      'angle': 26.666666666666668,
                      'prob': 0.8
                  }, {
                      'type': 'ColorTransform',
                      'magnitude': 0.2,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.8,
                      'prob': 0.8
                  }, {
                      'type': 'Solarize',
                      'thr': 56.888888888888886,
                      'prob': 0.8
                  }],
                  [{
                      'type': 'Sharpness',
                      'magnitude': 0.7,
                      'prob': 0.4
                  }, {
                      'type': 'Invert',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Shear',
                      'magnitude': 0.16666666666666666,
                      'prob': 0.6,
                      'direction': 'horizontal'
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.0,
                      'prob': 0.4
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.4
                  }, {
                      'type': 'Solarize',
                      'thr': 142.22222222222223,
                      'prob': 0.2
                  }],
                  [{
                      'type': 'Solarize',
                      'thr': 113.77777777777777,
                      'prob': 0.6
                  }, {
                      'type': 'AutoContrast',
                      'prob': 0.6
                  }],
                  [{
                      'type': 'Invert',
                      'prob': 0.6
                  }, {
                      'type': 'Equalize',
                      'prob': 1.0
                  }],
                  [{
                      'type': 'ColorTransform',
                      'magnitude': 0.4,
                      'prob': 0.6
                  }, {
                      'type': 'Contrast',
                      'magnitude': 0.8,
                      'prob': 1.0
                  }],
                  [{
                      'type': 'Equalize',
                      'prob': 0.8
                  }, {
                      'type': 'Equalize',
                      'prob': 0.6
                  }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=48,
    workers_per_gpu=1,
    train=dict(
        type='CustomDataset',
        data_prefix='data//camp01_base//train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='data//camp01_base//val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]),
    test=dict(
        type='CustomDataset',
        data_prefix='data//camp01_base//val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, -1)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ]))
evaluation = dict(interval=1, save_best='auto', metric='accuracy')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[10, 20])
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=10)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = './/pretrain//resnet18_8xb32_in1k_20210831-fbbb1da6.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
fp16 = dict(loss_scale='dynamic')
work_dir = './/work_dirs//camp1_basic'
gpu_ids = [0]
