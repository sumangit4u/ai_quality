"""
Model Export Script
Helps export trained models from Class 1 notebooks to production API format
"""

import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== Model Definitions ========================
# (Same as in api.py)

import torch.nn as nn
from torchvision.models import resnet18

class BaselineModel(nn.Module):
    """Version 1: Basic ResNet-18"""
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


class DropoutModel(nn.Module):
    """Version 2: ResNet-18 with Dropout"""
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.resnet = resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)


# ======================== Export Functions ========================

def export_model(
    model: nn.Module,
    output_dir: str,
    version: str,
    model_name: str,
    num_classes: int,
    metadata: dict = None
) -> bool:
    """
    Export a trained model to production format
    
    Args:
        model: Trained PyTorch model
        output_dir: Directory to save model
        version: Model version (e.g., 'v1.0', 'v2.0')
        model_name: Name of the model
        num_classes: Number of output classes
        metadata: Additional metadata to save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create version directory
        version_dir = Path(output_dir) / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_path = version_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"✓ Saved model weights: {model_path}")
        
        # Save model metadata
        default_metadata = {
            "model_name": model_name,
            "version": version,
            "num_classes": num_classes,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "exported_at": datetime.now().isoformat(),
            "framework": "pytorch",
            "pytorch_version": torch.__version__,
            "architecture": model.__class__.__name__
        }
        
        if metadata:
            default_metadata.update(metadata)
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(default_metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")
        
        # Save class names (if provided)
        if 'class_names' in metadata:
            classes_path = version_dir / "classes.json"
            with open(classes_path, 'w') as f:
                json.dump(metadata['class_names'], f, indent=2)
            logger.info(f"✓ Saved class names: {classes_path}")
        
        # Create README
        readme_content = f"""# {model_name} - Version {version}

## Model Information
- **Framework**: PyTorch
- **Architecture**: {model.__class__.__name__}
- **Parameters**: {default_metadata['num_parameters']:,}
- **Classes**: {num_classes}
- **Exported**: {default_metadata['exported_at']}

## Files
- `model.pth`: Model weights in PyTorch format
- `metadata.json`: Model configuration and metadata
- `classes.json`: Class names
- `README.md`: This file

## Usage in Production API

```python
from api import {model.__class__.__name__}

# Load model
model = {model.__class__.__name__}(num_classes={num_classes})
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()
```

## Training Information
{f"**Accuracy**: {metadata.get('accuracy', 'N/A')}" if metadata else ""}
{f"**Loss**: {metadata.get('loss', 'N/A')}" if metadata else ""}

## Performance Metrics
- **Inference Latency**: Measured during predictions
- **Model Size**: {Path(model_path).stat().st_size / 1024 / 1024:.2f} MB
"""
        
        readme_path = version_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"✓ Created README: {readme_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error exporting model: {str(e)}")
        return False


def load_and_verify_model(
    model_dir: str,
    model_class,
    num_classes: int
) -> bool:
    """
    Load and verify an exported model
    
    Args:
        model_dir: Directory containing exported model
        model_class: Model class to instantiate
        num_classes: Number of classes
    
    Returns:
        True if model loads successfully
    """
    try:
        device = torch.device("cpu")
        
        # Load model
        model = model_class(num_classes).to(device)
        model_path = Path(model_dir) / "model.pth"
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"✓ Model loaded successfully from {model_path}")
        
        # Test with dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128).to(device)
            output = model(dummy_input)
            logger.info(f"✓ Model output shape: {output.shape}")
            assert output.shape[1] == num_classes, f"Expected {num_classes} outputs, got {output.shape[1]}"
        
        logger.info(f"✓ Model verification successful!")
        return True
    
    except Exception as e:
        logger.error(f"✗ Error verifying model: {str(e)}")
        return False


# ======================== Main ========================

def main():
    parser = argparse.ArgumentParser(
        description="Export trained models from notebooks to production format"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models',
        help='Output directory for exported models (default: ./models)'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        required=True,
        choices=['v1.0', 'v2.0'],
        help='Model version to export as'
    )
    
    parser.add_argument(
        '--model-class',
        type=str,
        required=True,
        choices=['BaselineModel', 'DropoutModel'],
        help='Model class type'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        default=7,
        help='Number of output classes (default: 7)'
    )
    
    parser.add_argument(
        '--accuracy',
        type=float,
        help='Model accuracy (optional, for metadata)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify model after export'
    )
    
    parser.add_argument(
        '--class-names',
        nargs='+',
        default=['animal', 'name_board', 'other_vehicle', 'pedestrian', 'pothole', 'road_sign', 'speed_breaker'],
        help='Class names'
    )
    
    args = parser.parse_args()
    
    # ======================== Export Flow ========================
    
    print("\n" + "="*70)
    print("🚀 MODEL EXPORT UTILITY")
    print("="*70)
    
    # Validate inputs
    if args.model_path and not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return 1
    
    # Select model class
    model_classes = {
        'BaselineModel': BaselineModel,
        'DropoutModel': DropoutModel
    }
    model_class = model_classes[args.model_class]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If model path provided, load and export it
    if args.model_path:
        logger.info(f"Loading model from: {args.model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(args.num_classes).to(device)
        
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return 1
    else:
        # Use random model for demo
        logger.warning("No model path provided. Creating random model for demo.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_class(args.num_classes).to(device)
    
    model.eval()
    
    # Prepare metadata
    metadata = {
        'class_names': args.class_names,
    }
    if args.accuracy:
        metadata['accuracy'] = args.accuracy
    
    # Export model
    print(f"\n📦 Exporting {args.model_class} as {args.version}...")
    success = export_model(
        model=model,
        output_dir=args.output_dir,
        version=args.version,
        model_name=args.model_class,
        num_classes=args.num_classes,
        metadata=metadata
    )
    
    if not success:
        return 1
    
    # Verify if requested
    if args.verify:
        print(f"\n✓ Verifying exported model...")
        version_dir = Path(args.output_dir) / args.version
        verify_success = load_and_verify_model(
            str(version_dir),
            model_class,
            args.num_classes
        )
        if not verify_success:
            return 1
    
    print("\n" + "="*70)
    print("✅ EXPORT SUCCESSFUL!")
    print("="*70)
    print(f"\n📍 Models saved to: {Path(args.output_dir).absolute()}")
    print(f"   Ready for API deployment!")
    print("\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
