from .openlane_mbv4_300d_baseline import dataloader, lr_multiplier, optimizer, param_config, train
from .model_factory import create_llanetv1_model

param_config.backbone_type = 'mobilenetv4'
param_config.backbone_name = 'mobilenetv4_conv_small'
param_config.neck_type = 'FPN'
train.output_dir = 'output/llanetv1/300d/mbv4_fpn'
param_config.output_dir = train.output_dir

model = create_llanetv1_model(param_config)
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'
