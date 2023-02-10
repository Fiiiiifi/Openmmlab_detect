_base_ = 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='/input0/train',
        classes=classes,
        ann_file='/input0/train/train.json'),
    val=dict(
        img_prefix='/input0/val',
        classes=classes,
        ann_file='/input0/val/val.json'),
    test=dict(
        img_prefix='/input0/val',
        classes=classes,
        ann_file='/input0/val/val.json'))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
evaluation = dict(interval=1, metric='bbox', save_best='auto')

load_from = 'mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'