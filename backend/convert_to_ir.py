import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import os

def convert_to_ir():
    print("Starting model conversion to OpenVINO IR...")
    
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
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Save the model as TorchScript
    print("Converting model to TorchScript...")
    traced_model = torch.jit.trace(model, dummy_input)
    ts_path = output_dir / "dog_breeds.pt"
    traced_model.save(str(ts_path))
    
    print(f"Model saved as TorchScript at {ts_path}")
    print("You can now use openvino.runtime.Core().read_model() to load this model")
    print("\nCommand to use in Python:")
    print("from openvino.runtime import Core")
    print("core = Core()")
    print(f"model = core.read_model('{ts_path}')")
    print("compiled_model = core.compile_model(model, 'CPU')")

if __name__ == "__main__":
    convert_to_ir() 