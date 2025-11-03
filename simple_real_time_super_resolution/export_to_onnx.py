import os
import argparse
import yaml
import torch
from types import SimpleNamespace

# Import model constructor
from source.models.get_model import get_model


def dict_to_namespace(d):
    """Recursively convert nested dictionaries to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def parse_args():
    parser = argparse.ArgumentParser(description='Export trained Super-Resolution model to ONNX')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file used for training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (*.pt, preferably *_deploy.pt)')
    parser.add_argument('--onnx_out', type=str, required=True,
                        help='Output ONNX filename')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU if available')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version (default: 18)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(cfg_dict)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"=> Using device: {device}")

    # Initialize model
    print("=> Building model...")
    model = get_model(cfg, device)
    model.eval()

    # Load checkpoint
    print(f"=> Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Support checkpoints saved as dicts with or without 'state_dict'
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    print("=> Checkpoint loaded successfully!")

    # Dummy input (1080p)
    dummy_input = torch.randn(1, 3, 1080, 1920, device=device)

    # Export ONNX
    print(f"=> Exporting model to ONNX (opset {args.opset})...")
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_out,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['lr_input'],
        output_names=['sr_output'],
        dynamic_axes={
            'lr_input': {2: 'height', 3: 'width'},
            'sr_output': {2: 'height_out', 3: 'width_out'}
        }
    )

    print(f"\n✅ ONNX model exported successfully to:\n{args.onnx_out}")


if __name__ == "__main__":
    main()

