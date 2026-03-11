from .openlane_mbv4_1000d_baseline import dataloader, lr_multiplier, optimizer, param_config, train
from .model_factory import create_llanetv1_model

param_config.enable_temporal_model = True
param_config.temporal_loss_weight = 0.5
train.output_dir = 'output/llanetv1/1000d/mbv4_full'
param_config.output_dir = train.output_dir

model = create_llanetv1_model(param_config)
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'
