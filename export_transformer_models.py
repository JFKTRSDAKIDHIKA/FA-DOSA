"""
Export Transformer Models for FA-DOSA Testing

This script creates simplified Transformer architectures and exports them to ONNX format
for testing the enhanced FA-DOSA framework's Transformer support.

The models are simplified but contain the key operators found in real Transformers:
- LayerNormalization
- MatMul/Gemm (for attention and feed-forward)
- Gelu activation
- Add (for residual connections)
- Softmax (for attention)

Usage:
    python export_transformer_models.py
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np


class SimpleBERTLayer(nn.Module):
    """
    Simplified BERT-like transformer layer for testing.
    Contains the essential operations: LayerNorm, MatMul, Add, Gelu.
    """
    def __init__(self, hidden_size=768, intermediate_size=3072):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        
        # Simplified attention (just the output projection)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward network
        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # Self-attention block (simplified)
        attention_input = self.layernorm1(x)
        attention_output = self.attention_output(attention_input)
        x = x + attention_output  # Residual connection
        
        # Feed-forward block
        ff_input = self.layernorm2(x)
        intermediate_output = self.intermediate(ff_input)
        intermediate_output = self.gelu(intermediate_output)
        ff_output = self.output(intermediate_output)
        x = x + ff_output  # Residual connection
        
        return x


class SimpleViTBlock(nn.Module):
    """
    Simplified Vision Transformer block for testing.
    """
    def __init__(self, hidden_size=384, num_heads=6, mlp_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp_hidden_size = int(hidden_size * mlp_ratio)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Simplified multi-head attention (just output projection)
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        
        # MLP block
        self.mlp_fc1 = nn.Linear(hidden_size, self.mlp_hidden_size)
        self.mlp_fc2 = nn.Linear(self.mlp_hidden_size, hidden_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # Attention block
        normed = self.norm1(x)
        attention_out = self.attention_output(normed)
        x = x + attention_out
        
        # MLP block  
        normed = self.norm2(x)
        mlp_out = self.mlp_fc1(normed)
        mlp_out = self.gelu(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        x = x + mlp_out
        
        return x


class SimpleGPTBlock(nn.Module):
    """
    Simplified GPT-like decoder block for testing.
    """
    def __init__(self, hidden_size=512, intermediate_size=2048):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer normalization (GPT uses pre-norm)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Simplified attention
        self.attention_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward
        self.mlp_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.mlp_fc2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # Pre-norm attention
        normed = self.ln1(x)
        attention_out = self.attention_proj(normed)
        x = x + attention_out
        
        # Pre-norm MLP
        normed = self.ln2(x)
        mlp_out = self.mlp_fc1(normed)
        mlp_out = self.gelu(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        x = x + mlp_out
        
        return x


def export_bert_base():
    """Export a simplified BERT-base model."""
    print("Exporting BERT-base model...")
    
    # Create model
    model = nn.Sequential(
        SimpleBERTLayer(hidden_size=768, intermediate_size=3072),
        SimpleBERTLayer(hidden_size=768, intermediate_size=3072),
        SimpleBERTLayer(hidden_size=768, intermediate_size=3072),
    )
    model.eval()
    
    # Create dummy input: (batch_size=1, seq_len=512, hidden_size=768)
    dummy_input = torch.randn(1, 512, 768)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "bert_base.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("  ✓ bert_base.onnx exported successfully")


def export_vit_small():
    """Export a simplified Vision Transformer model."""
    print("Exporting ViT-small model...")
    
    # Create model
    model = nn.Sequential(
        SimpleViTBlock(hidden_size=384, num_heads=6),
        SimpleViTBlock(hidden_size=384, num_heads=6),
        SimpleViTBlock(hidden_size=384, num_heads=6),
        SimpleViTBlock(hidden_size=384, num_heads=6),
    )
    model.eval()
    
    # Create dummy input: (batch_size=1, num_patches=196, hidden_size=384)
    # For 224x224 image with 16x16 patches: 14*14 = 196 patches
    dummy_input = torch.randn(1, 196, 384)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "vit_small.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'num_patches'},
            'output': {0: 'batch_size', 1: 'num_patches'}
        }
    )
    print("  ✓ vit_small.onnx exported successfully")


def export_gpt2_small():
    """Export a simplified GPT-2 small model."""
    print("Exporting GPT-2 small model...")
    
    # Create model
    model = nn.Sequential(
        SimpleGPTBlock(hidden_size=512, intermediate_size=2048),
        SimpleGPTBlock(hidden_size=512, intermediate_size=2048),
        SimpleGPTBlock(hidden_size=512, intermediate_size=2048),
        SimpleGPTBlock(hidden_size=512, intermediate_size=2048),
    )
    model.eval()
    
    # Create dummy input: (batch_size=1, seq_len=256, hidden_size=512)
    dummy_input = torch.randn(1, 256, 512)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "gpt2_small.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("  ✓ gpt2_small.onnx exported successfully")


def export_distilbert():
    """Export a simplified DistilBERT model."""
    print("Exporting DistilBERT model...")
    
    # Create model (fewer layers than BERT)
    model = nn.Sequential(
        SimpleBERTLayer(hidden_size=768, intermediate_size=3072),
        SimpleBERTLayer(hidden_size=768, intermediate_size=3072),
    )
    model.eval()
    
    # Create dummy input: (batch_size=1, seq_len=512, hidden_size=768)
    dummy_input = torch.randn(1, 512, 768)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "distilbert.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("  ✓ distilbert.onnx exported successfully")


def main():
    """Export all Transformer models for FA-DOSA testing."""
    print("=" * 60)
    print("EXPORTING TRANSFORMER MODELS FOR FA-DOSA TESTING")
    print("=" * 60)
    
    try:
        export_bert_base()
        export_vit_small()
        export_gpt2_small()
        export_distilbert()
        
        print("\n" + "=" * 60)
        print("ALL TRANSFORMER MODELS EXPORTED SUCCESSFULLY")
        print("=" * 60)
        print("\nGenerated files:")
        print("  • bert_base.onnx      - BERT-Base architecture")
        print("  • vit_small.onnx      - Vision Transformer Small")
        print("  • gpt2_small.onnx     - GPT-2 Small")
        print("  • distilbert.onnx     - DistilBERT")
        print("\nThese models can now be used with run_experiments.py")
        print("to test FA-DOSA's enhanced Transformer support.")
        
    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 