"""Convert ONNX-zoo SqueezeNet 1.1 and VGG-16 (ImageNet, MXNet Gluon origin) to torchvision state dicts."""
import os, onnx, numpy as np, torch, torchvision
from onnx import numpy_helper

def onnx_params(path):
    m = onnx.load(path)
    return [(i.name, numpy_helper.to_array(i)) for i in m.graph.initializer], m

def to_tv(path, tv_model, out):
    params, m = onnx_params(path)
    # keep conv/linear weights & biases in graph order of use
    init = dict(params)
    order = []
    for node in m.graph.node:
        if node.op_type in ("Conv", "Gemm"):
            for inp in node.input[1:]:
                if inp in init: order.append(init[inp])
    sd = tv_model.state_dict()
    keys = [k for k in sd if k.endswith("weight") or k.endswith("bias")]
    assert len(keys) == len(order), (len(keys), len(order))
    new = {}
    for k, v in zip(keys, order):
        v = torch.from_numpy(np.array(v)).float()
        if v.shape != sd[k].shape:
            v = v.reshape(sd[k].shape)
        new[k] = v
    tv_model.load_state_dict(new)
    torch.save(tv_model.state_dict(), out)
    print("saved", out, len(keys), "tensors")

W = os.environ.get("DAX_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"))
to_tv(os.path.join(W, "squeezenet1.1-7.onnx"), torchvision.models.squeezenet1_1(), os.path.join(W, "squeezenet1_1_imagenet.pth"))
to_tv(os.path.join(W, "vgg16-7.onnx"), torchvision.models.vgg16(), os.path.join(W, "vgg16_imagenet.pth"))
