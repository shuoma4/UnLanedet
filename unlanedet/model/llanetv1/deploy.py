from copy import deepcopy

import torch


def prepare_qat(model, backend='fbgemm'):
    if not hasattr(torch, 'ao') or not hasattr(torch.ao, 'quantization'):
        raise RuntimeError('PyTorch quantization is not available in the current environment.')
    torch.backends.quantized.engine = backend
    model = deepcopy(model)
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model, inplace=False)


def convert_deploy(model):
    if not hasattr(torch, 'ao') or not hasattr(torch.ao, 'quantization'):
        raise RuntimeError('PyTorch quantization is not available in the current environment.')
    deploy_model = deepcopy(model).eval()
    return torch.ao.quantization.convert(deploy_model, inplace=False)


def export_onnx(model, sample_batch, output_path, opset_version=17):
    model = deepcopy(model).eval()
    torch.onnx.export(
        model,
        sample_batch,
        output_path,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['batch'],
        output_names=['predictions'],
    )
