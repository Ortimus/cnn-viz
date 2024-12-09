import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import graphviz
import pandas as pd

def create_dynamic_cnn_model(input_shape, num_conv_layers, filters_list, kernel_sizes, num_dense_layers, dense_units_list, num_classes):
    # Create input layer explicitly
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv2D(filters_list[0], kernel_sizes[0], activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Add additional convolutional layers
    for i in range(1, num_conv_layers):
        x = layers.Conv2D(filters_list[i], kernel_sizes[i], activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Dense layers
    for i in range(num_dense_layers - 1):
        x = layers.Dense(dense_units_list[i], activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def format_model_summary(model):
    """Convert model summary to a pandas DataFrame with activation info"""
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    
    # Parse the summary into a structured format
    layers_list = []
    dense_layers = len([layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)])
    current_dense = 0
    
    # Get input shape more robustly
    try:
        # Try different methods to get input shape
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
        elif hasattr(model, 'input'):
            input_shape = model.input.shape
        else:
            # Parse from the summary text
            input_line = stringlist[1]  # First line after Model summary
            if 'input_shape' in input_line.lower():
                input_shape = input_line.split(':')[-1].strip()
            else:
                input_shape = "unknown"
    except Exception as e:
        print(f"Error getting input shape: {e}")
        input_shape = "unknown"
        
    layers_list.append({
        'Layer Type': '(Input)',
        'Output Shape': str(input_shape),
        'Activation': 'None',
        'Parameters': 0
    })

    
    # Parse each layer from the model directly
    for layer in model.layers:
        layer_type = f"({layer.__class__.__name__})"
        
        if isinstance(layer, tf.keras.layers.Dense):
            current_dense += 1
            activation = "Softmax" if current_dense == dense_layers else "ReLU"
        elif isinstance(layer, tf.keras.layers.Conv2D):
            activation = "ReLU"
        else:
            activation = "None"
            
        # Get layer info from summary text, but fall back to shape helper if needed
        layer_info = next((line for line in stringlist[2:-4] 
                          if line.strip() and layer_type in line), '')
        if layer_info:
            try:
                parts = layer_info.strip().split()
                output_shape = ' '.join(parts[1:-1])
                params = int(parts[-1].replace(',', ''))
            except (IndexError, ValueError):
                # If parsing fails, use helper function
                shape = get_layer_shape(layer, 'output')
                output_shape = str(shape) if shape is not None else 'unknown'
                params = layer.count_params()
        else:
            # Use helper function if no summary text found
            shape = get_layer_shape(layer, 'output')
            output_shape = str(shape) if shape is not None else 'unknown'
            params = layer.count_params()
            
        layer_data = {
            'Layer Type': layer_type,
            'Output Shape': output_shape,
            'Activation': activation,
            'Parameters': params
        }
        layers_list.append(layer_data)
    
    # Add total parameters row
    total_params = model.count_params()
    layers_list.append({
        'Layer Type': 'Total',
        'Output Shape': '',
        'Activation': '',
        'Parameters': total_params
    })
    
    df = pd.DataFrame(layers_list)
    
    # Improve display format
    st.dataframe(
        df.style
        .format({
            'Parameters': '{:,}',
        })
        .set_properties(**{
            'text-align': 'left',
            'font-family': 'monospace',
        })
    )

def get_layer_shape(layer, shape_type='output'):
    """Helper function to get layer shape safely using Keras API"""
    st.write(f"\nGetting {shape_type} shape for layer: {layer.__class__.__name__}")
    try:
        config = layer.get_config()
        st.write(f"Config: {config}")
        
        if shape_type == 'input':
            st.write("Trying input shape methods...")
            # Try different ways to get input shape
            if hasattr(layer, 'input_shape'):
                st.write("Using input_shape attribute")
                return layer.input_shape
            elif hasattr(layer, '_build_input_shape'):
                st.write("Using _build_input_shape attribute")
                return layer._build_input_shape
            elif 'batch_input_shape' in config:
                st.write("Using batch_input_shape from config")
                return config['batch_input_shape']
            
            # Additional attempts for input shape
            try:
                st.write("Trying to get shape from layer build")
                if hasattr(layer, '_built_input_shape'):
                    return layer._built_input_shape
                elif hasattr(layer, '_inbound_nodes') and layer._inbound_nodes:
                    return layer._inbound_nodes[0].input_shapes[0]
            except Exception as e:
                st.write(f"Additional input shape attempts failed: {e}")
            
        else:  # output shape
            st.write("Trying output shape methods...")
            if isinstance(layer, tf.keras.layers.Dense):
                st.write("Dense layer - using units for output shape")
                return (None, config.get('units'))
            elif isinstance(layer, tf.keras.layers.Conv2D):
                st.write("Conv2D layer - calculating output shape")
                if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                    input_shape = layer.input_shape
                    filters = config.get('filters')
                    return (None, input_shape[1], input_shape[2], filters)
            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                st.write("MaxPooling2D layer - calculating output shape")
                if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                    input_shape = layer.input_shape
                    pool_size = config.get('pool_size', (2, 2))
                    return (None, input_shape[1]//pool_size[0], 
                           input_shape[2]//pool_size[1], input_shape[3])
            elif isinstance(layer, tf.keras.layers.Flatten):
                st.write("Flatten layer - calculating output shape")
                if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                    input_shape = layer.input_shape
                    if len(input_shape) == 4:
                        return (None, input_shape[1] * input_shape[2] * input_shape[3])
            
            # Try getting output shape directly from layer
            try:
                st.write("Trying direct output shape computation")
                if hasattr(layer, 'compute_output_shape') and layer.input_shape is not None:
                    return layer.compute_output_shape(layer.input_shape)
            except Exception as e:
                st.write(f"Output shape computation failed: {e}")
        
        st.write(f"No shape found for {layer.__class__.__name__}")
        return None
            
    except Exception as e:
        st.write(f"Error in get_layer_shape: {str(e)}")
        return None
    
    
def model_to_graphviz(model):
    try:
        st.write("Starting graph creation...")
        dot = graphviz.Digraph(engine='dot', format='png')
        
        # Make the graph wider and adjust spacing
        dot.attr(rankdir='TB',       
                nodesep='2.0',       # Horizontal space between nodes
                ranksep='1.5')       # Vertical space between ranks
        
        # Add graph-level attributes for better layout
        dot.graph_attr.update({
            'splines': 'ortho',
            'concentrate': 'true',
            'fontsize': '16',
            'size': '45,40',        # Wide and tall fixed size
            'margin': '0.5'
        })
        
        layer_colors = {
            'Input': '#E1F5E1',
            'Conv2D': '#E6F3FF',
            'MaxPooling2D': '#F0F7EA',
            'Flatten': '#FFF4E6',
            'Dense': '#F3E6FF',
            'ReLU': '#FFE6E6',
            'Softmax': '#E6FFEF',
            'Output': '#F5E1E1',
            'Calculation': '#FFF8DC'  # Light yellow for calculations
        }
        
        node_counter = 0
        
        # Handle input shape from model directly
        st.write("Getting input shape...")
        model_input = model.input_shape
        if isinstance(model_input, tuple):
            height, width = model_input[1:3]
            channels = model_input[3]
        else:
            height, width = model_input[0][1:3]
            channels = model_input[0][3]
        
        st.write(f"Input dimensions: {height}x{width}x{channels}")    
        channels_text = "RGB" if channels == 3 else "Grayscale"
        input_name = f'layer_{node_counter}'
        
        dot.node(input_name,
                f'Input\n{height}x{width}x{channels}\n{channels_text}',
                shape='box',
                style='filled',
                fillcolor=layer_colors['Input'],
                width='3.0',
                height='1.3')
        
        prev_node = input_name
        node_counter += 1
        
        st.write(f"\nTotal layers in model: {len(model.layers)}")
        dense_count = 0
        
        for layer in model.layers:
            st.write(f"\nProcessing layer {node_counter}: {layer.__class__.__name__}")
            layer_type = layer.__class__.__name__
            layer_name = f'layer_{node_counter}'
            calc_name = f'calc_{node_counter}'
            
            # Skip InputLayer since we handled input separately
            if isinstance(layer, tf.keras.layers.InputLayer):
                st.write("Skipping InputLayer")
                continue
                
            # Get shapes using helper method
            st.write("Getting shapes...")
            input_shape = get_layer_shape(layer, 'input')
            output_shape = get_layer_shape(layer, 'output')
            
            st.write(f"Input shape: {input_shape}")
            st.write(f"Output shape: {output_shape}")
            
            # Skip if we can't get valid shapes
            if input_shape is None or output_shape is None:
                st.write(f"Skipping layer {layer_type} due to missing shape information")
                continue
            
            # Default values
            calc_text = ""
            label = layer_type
            color = '#FFFFFF'
            add_activation = False
            activation_type = None
            
            st.write(f"Processing layer type: {layer_type}")
            
            if isinstance(layer, layers.Conv2D):
                calc_text = (
                    f'Input: {input_shape[1]}×{input_shape[2]}×{input_shape[3]}\n'
                    f'Kernel: {layer.kernel_size[0]}×{layer.kernel_size[1]}, Filters: {layer.filters}\n'
                    f'Parameters: {input_shape[3]}×{layer.kernel_size[0]}×{layer.kernel_size[1]}×{layer.filters} + {layer.filters}\n'
                    f'Output: {output_shape[1]}×{output_shape[2]}×{output_shape[3]}'
                )
                label = f'{layer_type}\n{layer.filters} filters\n{layer.kernel_size[0]}x{layer.kernel_size[1]}'
                color = layer_colors['Conv2D']
                add_activation = True
                activation_type = 'ReLU'
                
            elif isinstance(layer, layers.MaxPooling2D):
                calc_text = (
                    f'Input: {input_shape[1]}×{input_shape[2]}×{input_shape[3]}\n'
                    f'Pool size: 2×2, Stride: 2\n'
                    f'Output: {output_shape[1]}×{output_shape[2]}×{output_shape[3]}'
                )
                label = f'{layer_type}\n2x2'
                color = layer_colors['MaxPooling2D']
                add_activation = False
                
            elif isinstance(layer, layers.Flatten):
                calc_text = (
                    f'Input: {input_shape[1]}×{input_shape[2]}×{input_shape[3]}\n'
                    f'Output: {output_shape[1]} (flattened)'
                )
                label = f'{layer_type}'
                color = layer_colors['Flatten']
                add_activation = False
                
            elif isinstance(layer, layers.Dense):
                dense_count += 1
                calc_text = (
                    f'Input size: {input_shape[1]}\n'
                    f'Output size: {layer.units}\n'
                    f'Parameters: {input_shape[1]}×{layer.units} + {layer.units}'
                )
                label = f'{layer_type}\n{layer.units} units'
                color = layer_colors['Dense']
                if dense_count < sum(1 for l in model.layers if isinstance(l, layers.Dense)):
                    add_activation = True
                    activation_type = 'ReLU'
                else:
                    add_activation = True
                    activation_type = 'Softmax'
            
            st.write(f"Creating nodes for layer {layer_type}")
            
            # Create a subgraph to keep layer and its calculation at same rank
            with dot.subgraph(name=f'cluster_{node_counter}') as cluster:
                cluster.attr(rank='same', style='invis')  # Make cluster invisible
                
                # Add layer node
                cluster.node(layer_name, 
                           label,
                           shape='box',
                           style='filled',
                           fillcolor=color,
                           width='3.0',     # Made wider
                           height='1.3')
                
                # Add calculation node if there is calculation text
                if calc_text:
                    cluster.node(calc_name,
                               calc_text,
                               shape='note',
                               style='filled,dashed',
                               fillcolor=layer_colors['Calculation'],
                               fontname='Courier',
                               fontsize='14',
                               margin='0.2',
                               width='3.0',    # Made less wide
                               height='1.2')
                    
                    # Connect with dashed line without arrow
                    cluster.edge(layer_name, calc_name, 
                               style='dashed', 
                               constraint='false', 
                               color='#666666',
                               dir='none')
            
            # Connect main flow
            dot.edge(prev_node, layer_name, penwidth='2.0')
            
            st.write(f"Adding activation: {add_activation}")
            # Add activation if needed
            if add_activation and activation_type:
                activation_name = f'activation_{node_counter}'
                dot.node(activation_name,
                        activation_type,
                        shape='ellipse',
                        style='filled',
                        fillcolor=layer_colors[activation_type],
                        width='1.5',
                        height='0.8')
                
                dot.edge(layer_name, activation_name, penwidth='2.0')
                prev_node = activation_name
                
                # Add output layer after the last softmax
                if activation_type == 'Softmax':
                    output_name = f'output_{node_counter}'
                    dot.node(output_name,
                           f'Output\n{output_shape[-1]} classes',
                           shape='box',
                           style='filled',
                           fillcolor=layer_colors['Output'],
                           width='3.0',
                           height='1.3')
                    dot.edge(activation_name, output_name, penwidth='2.0')
            else:
                prev_node = layer_name
            
            node_counter += 1
            st.write(f"Finished processing layer {layer_type}")
        
        st.write("Graph creation completed")
        return dot
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        st.write(f"Error details: {error_details}")
        st.error(f"Visualization error: {str(e)}")
        return None

def main():
    st.title("Dynamic CNN Model Architecture")

    # Sidebar configurations
    st.sidebar.title("Configure CNN Model")

    # Common image sizes in computer vision
    image_size_options = {
        "16x16": (16, 16),
        "28x28": (28, 28),  # MNIST size
        "32x32": (32, 32),  # CIFAR-10 size
        "64x64": (64, 64),
        "96x96": (96, 96),
        "128x128": (128, 128),
        "224x224": (224, 224),  # ImageNet typical size
        "256x256": (256, 256)
    }

    # Input configuration
    st.sidebar.subheader("Input Configuration")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        image_size = st.selectbox(
            "Input Image Size",
            options=list(image_size_options.keys()),
            index=2
        )
    with col2:
        channels = st.selectbox(
            "Color Channels",
            options=["Grayscale (1)", "RGB (3)"],
            index=0
        )

    # Convert selections to numeric values
    height, width = image_size_options[image_size]
    num_channels = 1 if "Grayscale" in channels else 3
    input_shape = (height, width, num_channels)


    # Warning for large architectures
    if height * width > 128 * 128:
        st.sidebar.warning("⚠️ Large input sizes may result in higher computational requirements.")

    # Model configuration
    num_conv_layers = st.sidebar.slider("Number of Conv Layers", min_value=1, max_value=5, value=2)
    kernel_size_options = ['1x1', '3x3', '5x5', '7x7']

    filters_list = []
    kernel_sizes = []
    for i in range(num_conv_layers):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            filters = st.slider(f"Conv Layer {i+1} Filters", min_value=16, max_value=64, step=8, value=32)
            filters_list.append(filters)
        with col2:
            kernel_size = st.selectbox(f"Kernel Size {i+1}", kernel_size_options, index=1)
            kernel_sizes.append(tuple(map(int, kernel_size.split('x'))))

    num_dense_layers = st.sidebar.slider("Number of Dense Layers (including output)", min_value=1, max_value=3, value=2)
    dense_units_list = [st.sidebar.slider(f"Dense Layer {i+1} Units", min_value=32, max_value=128, step=16, value=64) 
                        for i in range(num_dense_layers - 1)]

    num_classes = st.sidebar.slider("Number of Output Classes", min_value=2, max_value=100, value=10)

    try:
        # Create the model
        cnn_model = create_dynamic_cnn_model(
            input_shape=input_shape,
            num_conv_layers=num_conv_layers,
            filters_list=filters_list,
            kernel_sizes=kernel_sizes,
            num_dense_layers=num_dense_layers,
            dense_units_list=dense_units_list,
            num_classes=num_classes
        )

        # Get total parameters
        total_params = cnn_model.count_params()
    
        # Display the model summary with parameter count
        st.subheader(f"Model Summary (Total Parameters: {total_params:,})")
        format_model_summary(cnn_model)

        # Display the model architecture
        st.subheader("Model Architecture")
        
        # Create graph 
        dot = model_to_graphviz(cnn_model)
        
    
        if dot is not None:
            # Display the graph
            st.graphviz_chart(dot)
            
            # Create buttons in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Download DOT file
                st.download_button(
                    label="Download Graph (DOT)",
                    data=dot.source,
                    file_name="cnn_architecture.dot",
                    mime="text/plain",
                )
            
            with col2:
                # Download PNG file
                try:
                    # Render the graph to PNG
                    png_data = dot.pipe(format='png')
                    
                    # Add download button for PNG
                    st.download_button(
                        label="Download Graph (PNG)",
                        data=png_data,
                        file_name="cnn_architecture.png",
                        mime="image/png",
                    )
                except Exception as e:
                    st.error(f"Could not create PNG: {str(e)}")


            # Add help text in sidebar
            with st.sidebar.expander("ℹ️ About Input Sizes"):
                st.write("""
                - **16x16**: Tiny images, good for simple patterns
                - **28x28**: MNIST digit classification size
                - **32x32**: CIFAR-10 dataset size
                - **64x64**: Good balance for medium complexity
                - **96x96**: Higher detail, moderate memory usage
                - **128x128**: Detailed images, higher memory usage
                - **224x224**: Standard ImageNet size
                - **256x256**: High resolution, requires more computing power
                
                Choose based on your:
                - Dataset characteristics
                - Available computational resources
                - Required level of detail
                """)
        else:
            st.error("Failed to create visualization")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()