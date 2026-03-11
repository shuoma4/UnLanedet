from .openlane_mbv4_300d_full import dataloader, lr_multiplier, model, optimizer, param_config, train

param_config.deploy_cfg = dict(enable_qat=True, enable_onnx_export=True, backend='fbgemm')
train.output_dir = 'output/llanetv1/300d/mbv4_qat'
param_config.output_dir = train.output_dir
dataloader.evaluator.output_dir = f'{train.output_dir}/eval_results'
