import onnx
from onnx import shape_inference
from collections import defaultdict
from fa_dosa_demo import ComputationGraph

def safe_get_dim_value(dim, default=1):
    """
    Safely extract dimension value, handling dynamic dimensions.
    
    Args:
        dim: ONNX dimension object
        default: Default value if dimension is dynamic or invalid
        
    Returns:
        Integer dimension value
    """
    if hasattr(dim, 'dim_value') and dim.dim_value > 0:
        return dim.dim_value
    elif hasattr(dim, 'dim_param') and dim.dim_param:
        # Dynamic dimension, use default
        return default
    else:
        return default

def extract_dims_from_shape(output_shape, op_type):
    """
    Extract FA-DOSA dimensions from ONNX tensor shape based on operation type.
    
    Args:
        output_shape: ONNX tensor shape dimensions
        op_type: Operation type for context-aware extraction
        
    Returns:
        Dictionary of FA-DOSA dimensions
    """
    layer_dims = {}
    
    # Handle different tensor ranks
    if len(output_shape.dim) == 4:  # Standard NCHW format
        layer_dims['N'] = safe_get_dim_value(output_shape.dim[0], 1)
        layer_dims['K'] = safe_get_dim_value(output_shape.dim[1])
        layer_dims['P'] = safe_get_dim_value(output_shape.dim[2])
        layer_dims['Q'] = safe_get_dim_value(output_shape.dim[3])
    elif len(output_shape.dim) == 2:  # After flatten/fully connected
        layer_dims['N'] = safe_get_dim_value(output_shape.dim[0], 1)
        layer_dims['K'] = safe_get_dim_value(output_shape.dim[1])
        layer_dims['P'] = 1  # Flattened
        layer_dims['Q'] = 1  # Flattened
    elif len(output_shape.dim) == 1:  # 1D output
        layer_dims['N'] = 1
        layer_dims['K'] = safe_get_dim_value(output_shape.dim[0])
        layer_dims['P'] = 1
        layer_dims['Q'] = 1
    else:
        # Fallback for other ranks
        layer_dims['N'] = 1
        layer_dims['K'] = 64  # Default channel count
        layer_dims['P'] = 1
        layer_dims['Q'] = 1
    
    return layer_dims

def parse_onnx_to_graph(onnx_model_path: str) -> ComputationGraph:
    """
    Parses an ONNX model and converts it into a ComputationGraph object.
    Enhanced to support ResNet-18 and complex operators.

    Args:
        onnx_model_path: The file path to the ONNX model.

    Returns:
        A populated ComputationGraph object.
    """
    model = onnx.load(onnx_model_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph

    graph_nodes = ComputationGraph()
    tensor_shapes = {t.name: t for t in graph.value_info}
    tensor_shapes.update({t.name: t for t in graph.input})
    tensor_shapes.update({t.name: t for t in graph.output})
    
    initializers = {i.name: i for i in graph.initializer}
    node_to_producer = {output: node.name for node in graph.node for output in node.output}

    for node in graph.node:
        try:
            # Get output tensor shape
            if node.output[0] not in tensor_shapes:
                print(f"Warning: No shape info for {node.name} output {node.output[0]}. Using defaults.")
                layer_dims = {'N': 1, 'K': 64, 'P': 1, 'Q': 1, 'C': 64}
            else:
                output_shape = tensor_shapes[node.output[0]].type.tensor_type.shape
                layer_dims = extract_dims_from_shape(output_shape, node.op_type)

            # Handle different operation types
            if node.op_type == 'Conv':
                # Extract convolution-specific dimensions
                if len(node.input) > 1 and node.input[1] in initializers:
                    weight_tensor = initializers[node.input[1]]
                    layer_dims['C'] = weight_tensor.dims[1]  # Input channels
                    layer_dims['R'] = weight_tensor.dims[2]  # Kernel height
                    layer_dims['S'] = weight_tensor.dims[3]  # Kernel width
                else:
                    # Fallback if weight info not available
                    layer_dims['C'] = layer_dims['K']  # Assume same as output channels
                    layer_dims['R'] = 3  # Default 3x3 kernel
                    layer_dims['S'] = 3
                    
            elif node.op_type in ['Relu', 'LeakyRelu', 'Sigmoid', 'Tanh']:
                # For activation layers, input channels = output channels
                layer_dims['C'] = layer_dims['K']
                
            elif node.op_type == 'BatchNormalization':
                # BatchNorm preserves tensor shape
                layer_dims['C'] = layer_dims['K']
                
            elif node.op_type == 'MaxPool':
                # MaxPool preserves channels, may change spatial dimensions
                layer_dims['C'] = layer_dims['K']
                # R, S represent pool kernel size (default 2x2)
                layer_dims['R'] = 2
                layer_dims['S'] = 2
                
            elif node.op_type == 'Add':
                # Element-wise addition for residual connections
                layer_dims['C'] = layer_dims['K']
                
            elif node.op_type == 'GlobalAveragePool':
                # Global average pooling reduces spatial dimensions to 1x1
                layer_dims['C'] = layer_dims['K']
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type in ['Flatten', 'Reshape']:
                # Flattening operations
                layer_dims['C'] = layer_dims['K']
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type in ['Gemm', 'MatMul']:
                # Fully connected layers
                layer_dims['C'] = layer_dims['K']  # Will be overridden if weight info available
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
                # Try to get input dimension from weights
                if len(node.input) > 1 and node.input[1] in initializers:
                    weight_tensor = initializers[node.input[1]]
                    if len(weight_tensor.dims) >= 2:
                        layer_dims['C'] = weight_tensor.dims[0]  # Input features
                        
            else:
                # Default case for unsupported operations
                print(f"Warning: Unsupported op_type '{node.op_type}' encountered at node '{node.name}'. "
                      f"Skipping dimension parsing but preserving graph connectivity.")
                layer_dims['C'] = layer_dims.get('K', 64)
                # Set default values for missing dimensions
                for dim in ['R', 'S']:
                    if dim not in layer_dims:
                        layer_dims[dim] = 1

            # Add layer to graph
            graph_nodes.add_layer(node.name, layer_dims, node.op_type)

        except Exception as e:
            print(f"Error processing node {node.name} ({node.op_type}): {e}")
            # Add with minimal dimensions to maintain connectivity
            layer_dims = {'N': 1, 'K': 64, 'P': 1, 'Q': 1, 'C': 64, 'R': 1, 'S': 1}
            graph_nodes.add_layer(node.name, layer_dims, node.op_type)

        # Add edges based on input/output connections
        for input_tensor_name in node.input:
            if input_tensor_name in node_to_producer:
                producer_node_name = node_to_producer[input_tensor_name]
                graph_nodes.add_edge(producer_node_name, node.name)

    # Enhanced fusion heuristics for ResNet patterns
    
    # 1. Conv -> ReLU pattern (original)
    for edge in graph_nodes.edges:
        src_node_name, dest_node_name = edge
        src_node = next((n for n in graph.node if n.name == src_node_name), None)
        dest_node = next((n for n in graph.node if n.name == dest_node_name), None)
        
        if src_node and dest_node and src_node.op_type == 'Conv' and dest_node.op_type == 'Relu':
            graph_nodes.add_fusion_group([src_node_name, dest_node_name])
    
    # 2. Conv -> BatchNorm -> ReLU pattern (ResNet-specific)
    node_dict = {n.name: n for n in graph.node}
    
    for node in graph.node:
        if node.op_type == 'Conv':
            conv_name = node.name
            
            # Find direct successors of this Conv
            conv_successors = [dest for src, dest in graph_nodes.edges if src == conv_name]
            
            for bn_candidate in conv_successors:
                if bn_candidate in node_dict and node_dict[bn_candidate].op_type == 'BatchNormalization':
                    bn_name = bn_candidate
                    
                    # Find successors of BatchNorm
                    bn_successors = [dest for src, dest in graph_nodes.edges if src == bn_name]
                    
                    for relu_candidate in bn_successors:
                        if relu_candidate in node_dict and node_dict[relu_candidate].op_type == 'Relu':
                            relu_name = relu_candidate
                            
                            # Create Conv -> BatchNorm -> ReLU fusion group
                            fusion_group = [conv_name, bn_name, relu_name]
                            graph_nodes.add_fusion_group(fusion_group)
                            print(f"Found Conv->BN->ReLU pattern: {fusion_group}")
                            break
                    break
            
    return graph_nodes

def main():
    """Example usage of the enhanced ONNX frontend."""
    onnx_model_file = "resnet18.onnx"
    try:
        computation_graph = parse_onnx_to_graph(onnx_model_file)
        print("Successfully parsed ONNX model into ComputationGraph.")
        print(f"\nTotal Layers: {len(computation_graph.layers)}")
        print(f"Total Edges: {len(computation_graph.edges)}")
        print(f"Total Fusion Groups: {len(computation_graph.fusion_groups)}")
        
        # Show first few layers as example
        print("\nFirst 5 Layers:")
        for i, (name, info) in enumerate(computation_graph.layers.items()):
            if i >= 5:
                break
            print(f"  - {name} ({info['type']}): {info['dims']}")
            
        print(f"\nFirst 5 Fusion Groups:")
        for i, group in enumerate(computation_graph.fusion_groups):
            if i >= 5:
                break
            print(f"  - {group}")
            
    except FileNotFoundError:
        print(f"Error: The model file '{onnx_model_file}' was not found.")
        print("Please run 'python export_resnet.py' first to generate it.")
    except Exception as e:
        print(f"An error occurred while parsing the ONNX model: {e}")

if __name__ == "__main__":
    main() 