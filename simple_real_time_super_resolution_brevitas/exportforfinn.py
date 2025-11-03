import torch
import torch.nn as nn
import importlib.util
from brevitas.export import export_onnx_qcdq


# ✅ Replacement for nn.PixelShuffle that is ONNX/FINN friendly
class PixelShuffleONNX(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        r = self.upscale_factor
        b, c, h, w = x.size()
        out_c = c // (r * r)
        # Reshape + permute = same as pixel shuffle
        x = x.view(b, r, r, out_c, h, w)
        x = x.permute(0, 3, 4, 1, 5, 2)  # [B, C, H, r, W, r]
        x = x.reshape(b, out_c, h * r, w * r)
        return x


# ✅ Helper to replace all nn.PixelShuffle modules recursively
def replace_pixelshuffle(module):
    for name, child in module.named_children():
        if isinstance(child, nn.PixelShuffle):
            print(f"🔸 Replacing PixelShuffle (upscale={child.upscale_factor})")
            setattr(module, name, PixelShuffleONNX(child.upscale_factor))
        else:
            replace_pixelshuffle(child)


def load_model(model_py, model_class, weight_path, device="cpu"):
    spec = importlib.util.spec_from_file_location("model_module", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, model_class)()
    ckpt = torch.load(weight_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model.to(device)


def main():
    model_py = "./source/models/QuantplainRepConv.py"
    model_class = "QuantPlainRepConv"
    weight_path = "./experiments/QAT_X2_Brevitas/Q-2025-1101-2053/models/model_x2_best_submission_deploy.pt"
    output_path = "lrsrn_finn_qat.onnx"
    input_size = (1, 3, 1080, 1920)

    print("🔹 Loading model...")
    model = load_model(model_py, model_class, weight_path)

    # Replace PixelShuffle layers before export
    print("🔹 Replacing PixelShuffle with ONNX-compatible version...")
    replace_pixelshuffle(model)

    dummy_input = torch.randn(input_size)
    print(f"🔹 Exporting quantized model to {output_path} ...")

    export_onnx_qcdq(model, args=dummy_input, export_path=output_path, opset_version=13)

    print("✅ Export complete.")


if __name__ == "__main__":
    main()

