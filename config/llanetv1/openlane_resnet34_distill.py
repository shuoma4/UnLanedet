from .openlane_mbv4_300d_full import dataloader, lr_multiplier, optimizer, param_config, train
from .model_factory import create_llanetv1_model

param_config.distill_cfg = dict(
    enable=True,
    teacher_cfg_module='config.llanetv1.openlane_resnet34_teacher',
    teacher_checkpoint='',
    feature_weight=1.0,
    logits_weight=0.5,
    temperature=4.0,
    feature_pairs=[
        dict(student_idx=0, teacher_idx=0, student_channels=64, teacher_channels=64),
        dict(student_idx=1, teacher_idx=1, student_channels=64, teacher_channels=64),
        dict(student_idx=2, teacher_idx=2, student_channels=64, teacher_channels=64),
    ],
)
train.output_dir = 'output/llanetv1/300d/mbv4_distill'
param_config.output_dir = train.output_dir

model = create_llanetv1_model(param_config)
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'
