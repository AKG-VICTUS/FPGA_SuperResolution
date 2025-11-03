# FPGA-Based Lightweight Super Resolution Neural Network Deployment

## Project Overview

This project focuses on training and deploying a lightweight super-resolution neural network model on FPGA platforms, with a comparative study of two FPGA-targeted deployment frameworks: FINN and Vitis AI. The objective is to enable real-time super-resolution inference on edge devices, particularly for resource-constrained environments such as endoscopic imaging applications.

**Model Used:** LRSRNN (Lightweight Real-time Super Resolution Neural Network) from CVPR Workshop 2023

**Target Platforms:** Xilinx FPGA boards (KV260 / Zynq UltraScale+)

**Deployment Frameworks:** FINN and Vitis AI

## Project Status

**Current Status:** Incomplete - Model preparation completed; FPGA deployment incomplete due to framework integration challenges.

### Completed Stages

- ✅ Model cloning and repository setup
- ✅ Dataset preparation and download
- ✅ Dependency installation
- ✅ FP32 baseline model training
- ✅ Baseline validation on test dataset
- ✅ ONNX export for Vitis AI
- ✅ Quantization-Aware Training (QAT) with Brevitas
- ✅ QONNX export for FINN
- ✅ FINN and Vitis AI framework cloning and dependency installation

### Incomplete Stages

- ❌ FPGA binary/artifact generation
- ❌ Runtime testing on FPGA hardware
- ❌ Comparative performance analysis

## Workflow Overview

### Phase 1: Model Training and Preparation

#### 1.1 Baseline Model Training (FP32)

The LRSRNN model was trained using full-precision (32-bit floating-point) parameters following the official repository instructions. The baseline training reproduced the reference results without the fine-tuning stage mentioned in the original paper.

**Outputs:**
- Trained FP32 baseline model checkpoint
- Validation metrics (PSNR/SSIM scores on test dataset)

#### 1.2 Baseline Validation

The trained FP32 model was validated against the test dataset to establish baseline performance metrics. This validation served as the reference point for comparing quantized models and ensuring no degradation in accuracy due to quantization.

### Phase 2: Model Export and Quantization

#### 2.1 ONNX Export for Vitis AI

A Python script was developed and executed to export the FP32 baseline model to ONNX format, compatible with the Vitis AI quantization and compilation pipeline.

**Script Purpose:** Convert PyTorch model → ONNX intermediate representation

**Outputs:**
- `model.onnx` (FP32 baseline in ONNX format)

#### 2.2 Quantization-Aware Training (QAT) with Brevitas

The baseline model was copied and updated with quantized layers using the Brevitas quantization framework. Training and inference scripts were updated to accommodate quantized operations.

**Process:**
1. Replace baseline model layers with Brevitas quantized modules
2. Update training loop for QAT
3. Update inference test scripts
4. Perform QAT on the model
5. Validate quantized model on test dataset

**Outputs:**
- QAT-trained quantized model checkpoint
- Quantization validation metrics (PSNR/SSIM)

#### 2.3 QONNX Export for FINN

A Python script was created to export the QAT-trained quantized model to QONNX format (Quantized ONNX), which is compatible with the FINN framework.

**Script Purpose:** Convert Brevitas-quantized PyTorch model → QONNX intermediate representation

**Outputs:**
- `model_quantized.qonnx` (Quantized model in QONNX format)

### Phase 3: FPGA Deployment Frameworks

#### 3.1 FINN Framework Integration

**Objective:** Compile quantized model to streaming dataflow architecture on FPGA

**Process:**
1. Clone FINN repository to local machine
2. Install FINN dependencies
3. Configure IP generation parameters
4. Generate FINN-compatible IP cores

**Challenges Encountered:**
- IP generation errors during synthesis
- Configuration mismatches between model and FINN constraints
- Missing or incompatible dependencies during IP generation workflow

**Status:** Framework setup completed; IP generation incomplete due to errors

#### 3.2 Vitis AI Framework Integration

**Objective:** Compile quantized model to DPU (Deep Learning Processing Unit) architecture on FPGA

**Process:**
1. Clone Vitis AI repository to local machine
2. Install Vitis AI dependencies
3. Execute quantization workflow
4. Generate FPGA binaries (xclbin files)
5. Generate runtime inference files

**Challenges Encountered:**
- Quantization workflow could not complete despite following official documentation
- Framework configuration issues
- Potential incompatibilities between Brevitas quantization format and Vitis AI quantizer expectations

**Status:** Framework setup completed; quantization workflow incomplete

## Directory Structure

```
FPGA_SR_Project/
├── README.md
├── models/
│   ├── baseline_fp32/
│   │   └── checkpoint.pth
│   ├── quantized_qat/
│   │   └── checkpoint.pth
│   └── exports/
│       ├── model.onnx
│       └── model_quantized.qonnx
├── scripts/
│   ├── train.py
│   ├── validate.py
│   ├── export_onnx.py
│   ├── qat_train.py
│   ├── export_qonnx.py
│   └── fpga_inference.py
├── datasets/
│   ├── train/
│   ├── val/
│   └── test/
├── finn/
│   ├── build.py
│   ├── config.yml
│   └── build_logs/
├── vitis_ai/
│   ├── quantize.py
│   ├── compile.py
│   └── build_logs/
└── results/
    ├── baseline_metrics.csv
    ├── quantized_metrics.csv
    └── comparison_analysis.md
```

## Model Details

### LRSRNN Architecture

- **Framework:** PyTorch
- **Task:** Image Super-Resolution (2x upscaling)
- **Input Resolution:** Typically 1920×1080 pixels
- **Output Resolution:** 2x upscaled (e.g.,  3840×2160)
- **Model Type:** Lightweight CNN-based architecture optimized for real-time inference

### Quantization Specifications

| Stage | Framework | Precision | Export Format | Purpose |
|-------|-----------|-----------|---------------|---------|
| Baseline | PyTorch | FP32 | ONNX | Vitis AI Pipeline |
| QAT | PyTorch + Brevitas | INT8 | QONNX | FINN Pipeline |

## Dependencies

### Training & Model Export
- PyTorch
- ONNX and onnx-simplifier
- Brevitas (for quantization)
- scikit-image (for PSNR/SSIM metrics)
- NumPy, Pandas

### FPGA Workflows
- FINN (with all dependencies)
- Vitis AI (with all dependencies)
- Xilinx Vitis HLS
- Xilinx Vivado Design Suite

## Key Findings and Insights

### Completed Work
1. **Model Training:** Successfully reproduced LRSRNN baseline without fine-tuning stage
2. **ONNX Export:** FP32 model exported to standard ONNX format without issues
3. **Quantization:** Brevitas QAT framework successfully applied to LRSRNN model
4. **Model Validation:** Quantized model validated; quantization-induced accuracy loss assessed

### Encountered Challenges

#### FINN Integration Issues
- IP generation failed due to: incompatible layer definitions, configuration constraints, or missing IP cores
- Root cause: Potential mismatch between LRSRNN architecture and FINN's supported layer types (expected CNN layers, but may require specific streaming-compatible variants)

#### Vitis AI Integration Issues
- Quantization workflow did not proceed beyond initial setup despite following official documentation
- Potential causes: 
  - Mismatch between Brevitas quantization format and Vitis AI's built-in quantizer expectations
  - Environment configuration issues or incompatible tool versions
  - Model-specific constraints not documented in official guides

## Recommendations for Future Work

### Short-term (Immediate)

1. **FINN Debugging:**
   - Review FINN error logs for specific layer/IP generation failures
   - Validate LRSRNN layer compatibility with FINN-supported layers
   - Consider simplifying model architecture to isolate problematic layers
   - Consult FINN community forums for similar integration issues

2. **Vitis AI Debugging:**
   - Investigate quantization workflow logs for specific failure points
   - Test with a simpler baseline model to isolate framework issues
   - Verify compatibility between Brevitas-exported QONNX and Vitis AI's expectations
   - Consider using Vitis AI's native quantization tool instead of Brevitas

### Medium-term

3. **Alternative Approaches:**
   - Evaluate HLS-based manual IP generation instead of automated frameworks
   - Consider intermediate frameworks (e.g., TensorFlow Lite for Arm + FPGA bridges)
   - Explore academic FPGA toolchains if open-source alternatives exist

4. **Model Optimization:**
   - Implement dynamic quantization strategies
   - Profile model to identify critical layers for precision preservation
   - Experiment with mixed-precision quantization

### Long-term

5. **Comparative Study:**
   - Once both frameworks are operational, benchmark latency, throughput, and power consumption
   - Document detailed comparison results
   - Provide recommendations for framework selection based on use case

## How to Reproduce This Work

### Prerequisites
- Linux environment (Ubuntu 20.04 or later recommended)
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)
- Xilinx tools installed (Vivado, HLS, Vitis)

### Step-by-step Setup

```bash
# 1. Clone LRSRNN repository to start from base repo
git clone https://github.com/Ganzooo/LRSRN.git
cd LRSRNN

#Or
# Clone this repo to start ahead with Dataset Prepared
git clone https://github.com/AKG-VICTUS/FPGA_SuperResolution.git

#This repo has two seperate folders corresponding to FP32 and QAT
FP32: simple_real_time_super_resolution
QAT: simple_real_time_super_resolution_brevitas

#The prepared dataset can be downloaded from the drive link
https://drive.google.com/drive/folders/1w7K8fRe_ukh5DwMLfcK-6pzv2yokwnuQ?usp=drive_link
Download this and paste it in root of subfolders

# 2. Create virtual environment (Seperately for both FP32 and QAT)
python3 -m venv sr_env
source sr_env/bin/activate

python3 -m venv sr_env_qat
source sr_env/bin/activate

# 3. Install training dependencies
pip install -r requirements.txt 

pip install brevitas torch torchvision (Only for QAT)

5. Train baseline model (cd simple_real_time_super_resolution)
python train.py --config ./configs/x2_final/repConv_x2_m4c32_relu_div2k_warmup_lr5e-4_b8_p384_normalize.yml --gpu_ids 0

(Training runs for approx 16 hours)

# 6. Validate baseline
python inference_time_test.py --config ./experiments/Val_X2_Best/PlainRepConv_x2_p384_m4_c32_relu_l1_adam_lr0.0005_e800_t2025-1031-1535/config.yml --weight ./experiments/Val_X2_Best/PlainRepConv_x2_p384_m4_c32_relu_l1_adam_lr0.0005_e800_t2025-1031-1535/models/model_x2_best_submission.pt --outPath ./experiments/Val_X2_Best/PlainRepConv_x2_p384_m4_c32_relu_l1_adam_lr0.0005_e800_t2025-1031-1535/Result --gpu_ids 0 --fp16 FP16

# 7. Export to ONNX
python export_onnx.py export_to_onnx.py --config ./experiments/Val_X2_Best/PlainRepConv_x2_p384_m4_c32_relu_l1_adam_lr0.0005_e800_t2025-1031-1535/config.yml --checkpoint ./experiments/Val_X2_Best/PlainRepConv_x2_p384_m4_c32_relu_l1_adam_lr0.0005_e800_t2025-1031-1535/models/model_x2_best_submission_deploy.pt --onnx_out ./exports/PlainRepConv_x2_1080p_to_4k.onnx  --opset 18

# 8. Train quantized model (QAT)
python train.py --config ./configs/x2_final/quant_repConv_x2_m4c32_relu_div2k_warmup_lr5e-4_b8_p384_normalize.yml --gpu_ids 0

# 9. Export to QONNX
python exportforfinn.py 

# 10. FINN workflow (to be completed)
cd finn
As described in https://finn.readthedocs.io/en/latest/command_line.html

# 11. Vitis AI workflow (to be completed)
cd vitis_ai
This is yet to be figured out. 
```

## Performance Metrics

### Baseline Model (FP32)
| Metric | Value | Dataset |
|--------|-------|---------|
| PSNR | [33.870] | DIV2K, Flickr2K |
| SSIM | [0.9314] | DIV2K, Flickr2K |
| Inference Time (GPU) | [11 ms] | Single image |

### Quantized Model (INT8)
| Metric | Value | Dataset | Degradation |
|--------|-------|---------|-------------|
| PSNR | [33.730] | DIV2K,Flickr2K | [0.004%] |
| SSIM | [0.9295] | DIV2K,FLickr2K | [0.002%] |


### FPGA Deployment (To be completed)
| Framework | Latency (ms) | Throughput (FPS) | Power (W) | Memory (MB) |
|-----------|--------------|------------------|-----------|-------------|
| FINN (Target) | [Pending] | [Pending] | [Pending] | [Pending] |
| Vitis AI (Target) | [Pending] | [Pending] | [Pending] | [Pending] |

## References

- LRSRNN Paper: [@InProceedings{Gankhuyag_2023_CVPR,
    author    = {Gankhuyag, Ganzorig and Yoon, Kihwan and Park, Jinman and Son, Haeng Seon and Min, Kyoungwon},
    title     = {Lightweight Real-Time Image Super-Resolution Network for 4K Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1746-1755}
}]
- FINN Framework: https://github.com/Xilinx/finn
- Vitis AI: https://github.com/Xilinx/Vitis-AI
- Brevitas: https://github.com/Xilinx/brevitas
- ONNX Standard: https://onnx.ai/

## Troubleshooting

### Common Issues

**Issue:** ONNX export fails with layer not supported error
- **Solution:** Check model architecture; ensure all custom layers have ONNX-compatible implementations

**Issue:** QONNX export encounters quantization format errors
- **Solution:** Verify Brevitas quantization layers are properly defined; check onnx_qonnx package version

**Issue:** FINN IP generation fails
- **Solution:** Review generated error logs; validate model layer types against FINN documentation; test with simplified model

**Issue:** Vitis AI quantization workflow hangs
- **Solution:** Check environment variables; verify FPGA platform definition file; consult Vitis AI documentation for platform-specific constraints

## Contact and Support

For questions regarding this project, refer to:
- LRSRNN Original Repository: [Insert URL]
- FINN Documentation: https://finn.readthedocs.io/
- Vitis AI Documentation: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html

## License

[Specify appropriate license based on components used]

---

**Last Updated:** November 3, 2025  
**Project Status:** In Progress - Awaiting FPGA framework resolution
