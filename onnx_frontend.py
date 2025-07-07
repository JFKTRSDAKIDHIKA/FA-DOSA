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

def extract_attribute_value(attributes, attr_name, default=None):
    """
    Extract attribute value from ONNX node attributes.
    
    Args:
        attributes: List of ONNX node attributes
        attr_name: Name of the attribute to extract
        default: Default value if attribute not found
        
    Returns:
        Attribute value or default
    """
    for attr in attributes:
        if attr.name == attr_name:
            if attr.type == onnx.AttributeProto.INT:
                return attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                return list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOAT:
                return attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                return attr.s.decode('utf-8')
    return default

def extract_dims_from_shape(output_shape, op_type):
    """
    Extract FA-DOSA dimensions from ONNX tensor shape based on operation type.
    Enhanced for comprehensive operator support.
    
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
    Industrial-grade ONNX parser for ResNet-18 and complex neural networks.
    Comprehensive support for all major operators and advanced fusion detection.

    Args:
        onnx_model_path: The file path to the ONNX model.

    Returns:
        A populated ComputationGraph object with accurate operator modeling.
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
    node_dict = {n.name: n for n in graph.node}

    print(f"Parsing ONNX model with {len(graph.node)} nodes...")

    for node in graph.node:
        try:
            # Get output tensor shape
            if node.output[0] not in tensor_shapes:
                print(f"Warning: No shape info for {node.name} output {node.output[0]}. Using defaults.")
                layer_dims = {'N': 1, 'K': 64, 'P': 1, 'Q': 1, 'C': 64}
            else:
                output_shape = tensor_shapes[node.output[0]].type.tensor_type.shape
                layer_dims = extract_dims_from_shape(output_shape, node.op_type)

            # === COMPREHENSIVE OPERATOR SUPPORT ===
            
            if node.op_type == 'Conv':
                # Convolution layer - extract kernel and channel information
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
                    
            elif node.op_type in ['Relu', 'LeakyRelu', 'Sigmoid', 'Tanh', 'Clip']:
                # Activation layers - preserve tensor shape
                layer_dims['C'] = layer_dims['K']
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type == 'BatchNormalization':
                # BatchNorm - infer dimensions from output shape (same as input)
                layer_dims['C'] = layer_dims['K']  # Input channels = output channels
                layer_dims['R'] = 1  # BatchNorm doesn't have spatial kernels
                layer_dims['S'] = 1
                
            elif node.op_type == 'MaxPool':
                # MaxPool - preserves channels, may change spatial dimensions
                layer_dims['C'] = layer_dims['K']
                # Extract kernel size from attributes
                kernel_shape = extract_attribute_value(node.attribute, 'kernel_shape', [2, 2])
                layer_dims['R'] = kernel_shape[0] if len(kernel_shape) >= 2 else 2
                layer_dims['S'] = kernel_shape[1] if len(kernel_shape) >= 2 else 2
                
            elif node.op_type == 'Add':
                # Element-wise addition - crucial for residual connections
                # Add has two inputs, infer dimensions from output shape
                layer_dims['C'] = layer_dims['K']
                layer_dims['R'] = 1  # Element-wise operation
                layer_dims['S'] = 1
                
            elif node.op_type == 'GlobalAveragePool':
                # Global average pooling - reduces spatial dimensions to 1x1
                layer_dims['C'] = layer_dims['K']
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1  # Global pooling doesn't have explicit kernel
                layer_dims['S'] = 1
                
            elif node.op_type in ['Flatten', 'Reshape']:
                # Flattening operations - reshape tensor
                layer_dims['C'] = layer_dims['K']  # May be overridden below
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type in ['Gemm', 'MatMul']:
                # Fully connected layers - matrix multiplication
                layer_dims['P'] = 1
                layer_dims['Q'] = 1
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
                # Try to get input dimension from weights
                if len(node.input) > 1 and node.input[1] in initializers:
                    weight_tensor = initializers[node.input[1]]
                    if len(weight_tensor.dims) >= 2:
                        layer_dims['C'] = weight_tensor.dims[0]  # Input features
                        # K is already set from output shape
                else:
                    layer_dims['C'] = layer_dims['K']  # Fallback
                    
            elif node.op_type in ['AveragePool', 'LpPool']:
                # Other pooling operations
                layer_dims['C'] = layer_dims['K']
                kernel_shape = extract_attribute_value(node.attribute, 'kernel_shape', [2, 2])
                layer_dims['R'] = kernel_shape[0] if len(kernel_shape) >= 2 else 2
                layer_dims['S'] = kernel_shape[1] if len(kernel_shape) >= 2 else 2
                
            elif node.op_type in ['Dropout', 'Identity']:
                # Pass-through operations
                layer_dims['C'] = layer_dims['K']
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type in ['Mul', 'Div', 'Sub']:
                # Element-wise arithmetic operations
                layer_dims['C'] = layer_dims['K']
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            elif node.op_type in ['Concat', 'Split']:
                # Tensor manipulation operations
                layer_dims['C'] = layer_dims['K']
                layer_dims['R'] = 1
                layer_dims['S'] = 1
                
            else:
                # Default case for unsupported operations - robust fallback
                print(f"Warning: Unsupported op_type '{node.op_type}' at node '{node.name}'. Preserving connectivity only.")
                layer_dims['C'] = layer_dims.get('K', 64)
                # Set default values for missing dimensions
                for dim in ['R', 'S']:
                    if dim not in layer_dims:
                        layer_dims[dim] = 1

            # Add layer to graph
            graph_nodes.add_layer(node.name, layer_dims, node.op_type)
            print(f"  Added {node.op_type} layer: {node.name} with dims {layer_dims}")

        except Exception as e:
            print(f"Error processing node {node.name} ({node.op_type}): {e}")
            # Add with minimal dimensions to maintain connectivity
            layer_dims = {'N': 1, 'K': 64, 'P': 1, 'Q': 1, 'C': 64, 'R': 1, 'S': 1}
            graph_nodes.add_layer(node.name, layer_dims, node.op_type)

    # === ENHANCED GRAPH CONNECTIVITY ===
    print("\nBuilding graph connectivity...")
    edge_count = 0
    
    for node in graph.node:
        # Handle multiple inputs correctly (important for Add operations)
        for input_tensor_name in node.input:
            # Skip initializers (weights, biases) - only connect actual computation nodes
            if input_tensor_name in node_to_producer and input_tensor_name not in initializers:
                producer_node_name = node_to_producer[input_tensor_name]
                graph_nodes.add_edge(producer_node_name, node.name)
                edge_count += 1
    
    print(f"  Added {edge_count} edges")

    # === ADVANCED FUSION HEURISTICS ===
    print("\nDetecting fusion patterns...")
    fusion_count = 0
    
    # Pattern 1: Conv -> BatchNorm -> ReLU (ResNet-specific)
    for node in graph.node:
        if node.op_type == 'Conv':
            conv_name = node.name
            
            # Find direct consumers of this Conv
            conv_consumers = graph_nodes.get_node_consumers(conv_name)
            
            for bn_candidate in conv_consumers:
                if bn_candidate in node_dict and node_dict[bn_candidate].op_type == 'BatchNormalization':
                    bn_name = bn_candidate
                    
                    # Check if BatchNorm has exactly one consumer
                    bn_consumers = graph_nodes.get_node_consumers(bn_name)
                    
                    for relu_candidate in bn_consumers:
                        if relu_candidate in node_dict and node_dict[relu_candidate].op_type in ['Relu', 'Clip']:
                            relu_name = relu_candidate
                            
                            # Verify this is a linear chain (no other consumers)
                            if len(conv_consumers) == 1 and len(bn_consumers) == 1:
                                fusion_group = [conv_name, bn_name, relu_name]
                                graph_nodes.add_fusion_group(fusion_group)
                                print(f"  Found Conv->BN->ReLU pattern: {' -> '.join(fusion_group)}")
                                fusion_count += 1
                            break
                    break
    
    # Pattern 2: Conv -> ReLU (simpler cases)
    for node in graph.node:
        if node.op_type == 'Conv':
            conv_name = node.name
            conv_consumers = graph_nodes.get_node_consumers(conv_name)
            
            # Only create Conv->ReLU if not already in a Conv->BN->ReLU group
            already_in_complex_fusion = any(
                conv_name in group and len(group) > 2 
                for group in graph_nodes.fusion_groups
            )
            
            if not already_in_complex_fusion:
                for relu_candidate in conv_consumers:
                    if (relu_candidate in node_dict and 
                        node_dict[relu_candidate].op_type in ['Relu', 'Clip'] and
                        len(conv_consumers) == 1):  # Ensure linear chain
                        
                        fusion_group = [conv_name, relu_candidate]
                        graph_nodes.add_fusion_group(fusion_group)
                        print(f"  Found Conv->ReLU pattern: {' -> '.join(fusion_group)}")
                        fusion_count += 1
                        break
    
    print(f"  Detected {fusion_count} fusion opportunities")
    print(f"\nParsing complete: {len(graph_nodes.layers)} layers, {len(graph_nodes.edges)} edges, {len(graph_nodes.fusion_groups)} fusion groups")
    
    return graph_nodes

def main():
    """Example usage of the enhanced industrial-grade ONNX frontend."""
    onnx_model_file = "resnet18.onnx"
    try:
        computation_graph = parse_onnx_to_graph(onnx_model_file)
        print("\n" + "="*60)
        print("ONNX PARSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total Layers: {len(computation_graph.layers)}")
        print(f"Total Edges: {len(computation_graph.edges)}")
        print(f"Total Fusion Groups: {len(computation_graph.fusion_groups)}")
        
        # Analyze operator types
        op_type_counts = {}
        for name, info in computation_graph.layers.items():
            op_type = info['type']
            op_type_counts[op_type] = op_type_counts.get(op_type, 0) + 1
        
        print(f"\nOperator Type Distribution:")
        for op_type, count in sorted(op_type_counts.items()):
            print(f"  {op_type}: {count}")
        
        # Show sample layers
        print(f"\nSample Layers (first 5):")
        for i, (name, info) in enumerate(computation_graph.layers.items()):
            if i >= 5:
                break
            print(f"  {name} ({info['type']}): {info['dims']}")
            
        print(f"\nSample Fusion Groups (first 5):")
        for i, group in enumerate(computation_graph.fusion_groups):
            if i >= 5:
                break
            group_types = [computation_graph.layers[layer]['type'] for layer in group]
            print(f"  {' -> '.join(group)} ({' -> '.join(group_types)})")
            
    except FileNotFoundError:
        print(f"Error: The model file '{onnx_model_file}' was not found.")
        print("Please run 'python export_resnet.py' first to generate it.")
    except Exception as e:
        print(f"An error occurred while parsing the ONNX model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 