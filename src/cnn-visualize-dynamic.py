import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import graphviz

# Function to create a dynamic CNN model with variable layers, kernel sizes, and output classes
def create_dynamic_cnn_model(input_shape, num_conv_layers, filters_list, kernel_sizes, num_dense_layers, dense_units_list, num_classes):
    model = models.Sequential()
    
    # Initial convolutional layer
    model.add(layers.Conv2D(filters_list[0], kernel_sizes[0], activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Add additional convolutional layers based on num_conv_layers
    for i in range(1, num_conv_layers):
        model.add(layers.Conv2D(filters_list[i], kernel_sizes[i], activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output to feed into fully connected layers
    model.add(layers.Flatten())
    
    # Add dense layers
    for i in range(num_dense_layers - 1):  # -1 because we add the output layer separately
        model.add(layers.Dense(dense_units_list[i], activation='relu'))
    
    # Output layer: num_classes units, softmax activation
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Function to create a Graphviz representation of the model with shallower boxes
def model_to_graphviz(model):
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB', nodesep='0.1', ranksep='0.3')
    
    def add_layer(layer, layer_id):
        layer_type = layer.__class__.__name__
        layer_shape = str(layer.output_shape)
        if isinstance(layer, layers.Conv2D):
            kernel_size = layer.kernel_size
            label = f'{layer_type} ({kernel_size[0]}x{kernel_size[1]})\\n{layer_shape}'
        else:
            label = f'{layer_type}\\n{layer_shape}'
        dot.node(str(layer_id), label, shape='box', style='filled,rounded', 
                 fillcolor='lightblue', width='2', height='0.7', 
                 fontname='Arial', fontsize='10')

    def connect_layers(prev_id, curr_id):
        dot.edge(str(prev_id), str(curr_id))

    for i, layer in enumerate(model.layers):
        add_layer(layer, i)
        if i > 0:
            connect_layers(i-1, i)

    return dot

# Streamlit app
st.title("Dynamic CNN Model Architecture")

# Streamlit sidebar sliders and dropdowns for dynamic control
st.sidebar.title("Configure CNN Model")
num_conv_layers = st.sidebar.slider("Number of Conv Layers", min_value=1, max_value=5, value=2)

# List of common kernel sizes
kernel_size_options = ['1x1', '3x3', '5x5', '7x7']

filters_list = []
kernel_sizes = []
for i in range(num_conv_layers):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        filters = st.slider(f"Conv Layer {i+1} Filters", min_value=16, max_value=64, step=8, value=32)
        filters_list.append(filters)
    with col2:
        kernel_size = st.selectbox(f"Kernel Size {i+1}", kernel_size_options, index=1)  # Default to 3x3
        kernel_sizes.append(tuple(map(int, kernel_size.split('x'))))

num_dense_layers = st.sidebar.slider("Number of Dense Layers (including output)", min_value=1, max_value=3, value=2)
dense_units_list = [st.sidebar.slider(f"Dense Layer {i+1} Units", min_value=32, max_value=128, step=16, value=64) for i in range(num_dense_layers - 1)]  # -1 because the output layer is handled separately

num_classes = st.sidebar.slider("Number of Output Classes", min_value=2, max_value=100, value=10)

# Use a larger input size to ensure enough dimensions
input_shape = (64, 64, 1)  # Increased input size

# Create the model based on the current settings
cnn_model = create_dynamic_cnn_model(input_shape=input_shape, num_conv_layers=num_conv_layers, 
                                     filters_list=filters_list, kernel_sizes=kernel_sizes,
                                     num_dense_layers=num_dense_layers, 
                                     dense_units_list=dense_units_list, num_classes=num_classes)

# Display the model summary
st.subheader("Model Summary")
model_summary = []
cnn_model.summary(print_fn=lambda x: model_summary.append(x))
st.text("\n".join(model_summary))

# Create and display the Graphviz representation
st.subheader("Model Architecture")
dot = model_to_graphviz(cnn_model)
st.graphviz_chart(dot, use_container_width=True)

# Add some spacing
st.write("")
st.write("")

# Explanation of the model
st.subheader("Model Explanation")
st.write(f"""
This Convolutional Neural Network (CNN) model is dynamically generated based on your input parameters. 
Here's a breakdown of its structure:

1. It starts with an input layer that accepts images of size 64x64 pixels with 1 channel (grayscale).
2. The model then has {num_conv_layers} convolutional layer(s), each followed by a max pooling layer. 
3. Each convolutional layer uses a configurable kernel size and ReLU activation function.
4. After the convolutional layers, the output is flattened.
5. There are {num_dense_layers - 1} dense (fully connected) layer(s) with ReLU activation.
6. Finally, there's an output dense layer with {num_classes} units and softmax activation, suitable for a {num_classes}-class classification problem.

You can adjust the model's complexity using the sliders and dropdowns in the sidebar. Experiment with different configurations to see how they affect the model's architecture!
""")
