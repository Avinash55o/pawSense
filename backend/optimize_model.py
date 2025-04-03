import torch
import torch.nn as nn
import torchvision.models as models
from openvino.tools import mo
from openvino.runtime import Core
import os
from pathlib import Path

def optimize_model():
    print("Starting model optimization with OpenVINO...")
    
    # Create output directory if it doesn't exist
    output_dir = Path('models/breed_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the trained model
    print("Loading trained PyTorch model...")
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 120)  # 120 breeds
    model.load_state_dict(torch.load('models/breed_model/mobilenetv2_dogbreeds.pth'))
    model.eval()
    
    # Create dummy input for tracing
    print("Creating dummy input for model tracing...")
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    print("Exporting model to ONNX format...")
    onnx_path = output_dir / "dog_breeds.onnx"
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,              # Model input
        onnx_path,                # Output path
        export_params=True,       # Store trained parameter weights inside the model file
        opset_version=11,         # ONNX version to export the model to
        do_constant_folding=True, # Whether to execute constant folding for optimization
        input_names=['input'],    # Model's input names
        output_names=['output'],  # Model's output names
        dynamic_axes={            # Variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Convert to OpenVINO IR
    print("Converting ONNX model to OpenVINO IR format...")
    mo.convert_model(
        str(onnx_path),
        model_name="dog_breeds",
        output_dir=str(output_dir),
        compress_to_fp16=True,    # Enable FP16 compression
        input_shape=[1, 3, 224, 224],
        input_name='input',
        output_name='output'
    )
    
    # Verify the converted model
    print("Verifying converted model...")
    ie = Core()
    model = ie.read_model(str(output_dir / "dog_breeds.xml"))
    compiled_model = ie.compile_model(model, device_name="CPU")
    
    print("\nOptimization completed successfully!")
    print(f"ONNX model saved to: {onnx_path}")
    print(f"OpenVINO model saved to: {output_dir}/dog_breeds.xml")
    print(f"OpenVINO weights saved to: {output_dir}/dog_breeds.bin")

if __name__ == "__main__":
    optimize_model() 