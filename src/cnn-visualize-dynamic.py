import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import graphviz

def create_dynamic_cnn_model(input_shape, num_conv_layers, filters_list, kernel_sizes, num_dense_layers, dense_units_list, num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(filters_list[0], kernel_sizes[0], activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    for i in range(1, num_conv_layers):
        model.add(layers.Conv2D(filters_list[i], kernel_sizes[i], activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    for i in range(num_dense_layers - 1):
        model.add(layers.Dense(dense_units_list[i], activation='relu'))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def model_to_graphviz(model):
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', nodesep='0.2', ranksep='0.3')  # Updated as requested
    
    def calculate_output_shape(layer, input_shape):
        if isinstance(layer, layers.Conv2D):
            return (input_shape[0], input_shape[1], layer.filters)  # Same padding
        elif isinstance(layer, layers.MaxPooling2D):
            return (input_shape[0] // 2, input_shape[1] // 2, input_shape[2])
        elif isinstance(layer, layers.Flatten):
            return (input_shape[0] * input_shape[1] * input_shape[2],)
        elif isinstance(layer, layers.Dense):
            return (layer.units,)
        else:
            return input_shape

    def calculate_params(layer, input_shape):
        if isinstance(layer, layers.Conv2D):
            return layer.kernel_size[0] * layer.kernel_size[1] * input_shape[2] * layer.filters + layer.filters
        elif isinstance(layer, layers.Dense):
            return input_shape[0] * layer.units + layer.units
        else:
            return 0

    def add_layer(layer, layer_id, input_shape):
        layer_type = layer.__class__.__name__
        layer_name = layer.name
        output_shape = calculate_output_shape(layer, input_shape)
        params = calculate_params(layer, input_shape)
        
        if isinstance(layer, layers.Conv2D):
            kernel_size = layer.kernel_size
            label = f'{layer_type} ({kernel_size[0]}x{kernel_size[1]})\\n{layer_name}\\n{output_shape}\\nParams: {params}'
            color = 'lightblue'
        elif isinstance(layer, layers.MaxPooling2D):
            label = f'{layer_type}\\n{layer_name}\\n{output_shape}'
            color = 'orange'
        elif isinstance(layer, layers.Dense):
            label = f'{layer_type}\\n{layer_name}\\n{output_shape}\\nParams: {params}'
            color = 'lightyellow'
        else:
            label = f'{layer_type}\\n{layer_name}\\n{output_shape}'
            color = 'white'
        
        return label, output_shape, color, params

    def edge_label(prev_shape, curr_shape, params):
        if len(prev_shape) == 3 and len(curr_shape) == 3:
            if prev_shape[0] != curr_shape[0] or prev_shape[1] != curr_shape[1]:
                return f'[Shape: {prev_shape[0]}/{2}={curr_shape[0]}]\\n[Params: {params}]'
            else:
                return f'[Shape: unchanged]\\n[Params: {params}]'
        elif len(prev_shape) == 3 and len(curr_shape) == 1:
            return f'[Flatten: {prev_shape[0]}*{prev_shape[1]}*{prev_shape[2]}={curr_shape[0]}]'
        elif len(prev_shape) == 1 and len(curr_shape) == 1:
            return f'[Params: {prev_shape[0]}*{curr_shape[0]}+{curr_shape[0]}={params}]'
        else:
            return ''

    # Add input shape node
    dot.node('input', f'Input\\n{model.input_shape[1:]}', shape='box', style='filled', fillcolor='white')

    prev_node = 'input'
    input_shape = model.input_shape[1:]
    for i, layer in enumerate(model.layers):
        with dot.subgraph(name=f'cluster_{i}') as c:
            c.attr(style='filled', color='lightgrey')
            layer_name = f'layer_{i}'
            try:
                layer_label, output_shape, color, params = add_layer(layer, i, input_shape)
            except Exception as e:
                layer_label = f"Layer {i}\\n(Error: {str(e)})"
                output_shape = input_shape
                color = 'red'
                params = 0
            c.node(layer_name, label=layer_label, shape='box', style='filled,rounded', 
                   fillcolor=color, width='2', height='0.8',
                   fontname='Arial', fontsize='10')
            
            if isinstance(layer, (layers.Conv2D, layers.Dense)) and i < len(model.layers) - 1:
                relu_name = f'relu_{i}'
                c.node(relu_name, "ReLU", shape='ellipse', style='filled', 
                       fillcolor='lightsalmon', width='1.5', height='0.5', 
                       fontname='Arial', fontsize='10')
                c.edge(layer_name, relu_name)
                last_node = relu_name
            else:
                last_node = layer_name
            
            edge_info = edge_label(input_shape, output_shape, params)
            # Adjusted edge label positioning
            dot.edge(prev_node, layer_name, label=edge_info, fontsize='8', 
                     labelangle='270', labeldistance='1.5', 
                     labeljust='l', tailport='s', headport='n')
            
            prev_node = last_node
            input_shape = output_shape

    return dot

st.title("Improved CNN Model Architecture Visualizer")

st.sidebar.title("Configure CNN Model")

# Input shape options
input_shape_options = {
    '16x16': (16, 16, 3),
    '32x32': (32, 32, 3),
    '64x64': (64, 64, 3),
    '128x128': (128, 128, 3),
    '224x224': (224, 224, 3)  # Common for transfer learning models
}
selected_input_shape = st.sidebar.selectbox("Select Input Shape", list(input_shape_options.keys()))
input_shape = input_shape_options[selected_input_shape]

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
dense_units_list = [st.sidebar.slider(f"Dense Layer {i+1} Units", min_value=32, max_value=128, step=16, value=64) for i in range(num_dense_layers - 1)]

num_classes = st.sidebar.slider("Number of Output Classes", min_value=2, max_value=100, value=10)

try:
    cnn_model = create_dynamic_cnn_model(input_shape=input_shape, num_conv_layers=num_conv_layers, 
                                         filters_list=filters_list, kernel_sizes=kernel_sizes,
                                         num_dense_layers=num_dense_layers, 
                                         dense_units_list=dense_units_list, num_classes=num_classes)

    st.subheader("Model Summary")
    model_summary = []
    cnn_model.summary(print_fn=lambda x: model_summary.append(x))
    st.text("\n".join(model_summary))

    st.subheader("Model Architecture")
    dot = model_to_graphviz(cnn_model)
    st.graphviz_chart(dot, use_container_width=True)

    st.write("")
    st.write("")

    st.subheader("Model Explanation")
    st.write(f"""
    This improved Convolutional Neural Network (CNN) model visualization includes:

    1. Input Layer: Accepts images of size {input_shape[0]}x{input_shape[1]} pixels with {input_shape[2]} channels.
    2. Convolutional Layers: {num_conv_layers} layer(s), each followed by ReLU activation and max pooling.
       - Kernel sizes and filter counts are configurable for each layer.
       - Output shapes and parameter counts are displayed for each layer.
    3. Flatten Layer: Converts the 2D feature maps to a 1D vector.
    4. Dense Layers: {num_dense_layers - 1} fully connected layer(s) with ReLU activation.
    5. Output Layer: Dense layer with {num_classes} units and softmax activation for {num_classes}-class classification.

    Color Coding:
    - White: Input layer
    - Light Blue: Convolutional layers
    - Orange: Max Pooling layers
    - Light Salmon: ReLU activation
    - Light Yellow: Dense layers

    Computation Details:
    - Shape changes are shown between layers, e.g., [Shape: 32/2=16] for max pooling.
    - Parameter calculations are displayed, e.g., [Params: 288] for convolutional layers.

    Formulas:
    1. Convolutional Layer Parameters: 
       (kernel_height * kernel_width * input_channels * num_filters) + num_filters
    2. Dense Layer Parameters: 
       (input_features * output_features) + output_features
    3. Max Pooling Shape: 
       new_height = height / pool_size, new_width = width / pool_size

    The visualization provides a detailed understanding of the model's architecture, shape changes, and parameter computations between layers.
    """)

except Exception as e:
    st.error(f"An error occurred while creating the model: {str(e)}")
    st.error("Please try adjusting the model parameters or check the console for more details.")
