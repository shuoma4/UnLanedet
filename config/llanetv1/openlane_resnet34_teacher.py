from .openlane_mbv4_300d_baseline import dataloader, get_config, lr_multiplier, optimizer, param_config, train
from .model_factory import create_llanetv1_model

param_config.backbone_type = 'resnet'
param_config.backbone_name = 'resnet50'
param_config.neck_type = 'FPN'
param_config.enable_global_semantic = False
param_config.use_data_driven_priors = False
param_config.assign_method = 'CLRNet'
param_config.enable_temporal_model = False
param_config.distill_cfg = dict(enable=False)
train.output_dir = 'output/llanetv1/300d/resnet50_teacher'
param_config.output_dir = train.output_dir

model = create_llanetv1_model(param_config)
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'
