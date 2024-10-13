import streamlit as st

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

import io
from PIL import Image


# Function to create a dynamic CNN model with variable layers
def create_dynamic_cnn_model(input_shape, num_layers, filters_list, dense_units):
    model = models.Sequential()
    
    # Initial convolutional layer
    model.add(layers.Conv2D(filters_list[0], (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Add additional convolutional layers based on num_layers
    for i in range(1, num_layers):
        model.add(layers.Conv2D(filters_list[i], (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output to feed into fully connected layers
    model.add(layers.Flatten())
    
    # Fully connected layer
    model.add(layers.Dense(dense_units, activation='relu'))
    
    # Output layer: 10 units (for 10 classes), softmax activation
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Streamlit sidebar sliders for dynamic control
st.sidebar.title("Configure CNN Model")
num_layers = st.sidebar.slider("Number of Conv Layers", min_value=2, max_value=5, value=2)
filters_list = [st.sidebar.slider(f"Conv Layer {i+1} Filters", min_value=16, max_value=64, step=8, value=32) for i in range(num_layers)]
dense_units = st.sidebar.slider("Dense Layer Units", min_value=32, max_value=128, step=16, value=64)

# Use a larger input size to ensure enough dimensions
input_shape = (64, 64, 1)  # Increased input size

# Create the model based on the current slider settings
cnn_model = create_dynamic_cnn_model(input_shape=input_shape, num_layers=num_layers, filters_list=filters_list, dense_units=dense_units)

# Display the model summary in the Streamlit app
st.title("Dynamic CNN Model Architecture")
model_summary = []
cnn_model.summary(print_fn=lambda x: model_summary.append(x))
st.text("\n".join(model_summary))

# Save the model architecture to an image and display it

# Create a buffer to save the image in-memory because we don't have access to the file system on streamlit cloud
buffer = io.BytesIO()

# plot_model(cnn_model, to_file='cnn_model_dynamic.png', show_shapes=True, show_layer_names=True)
plot_model(cnn_model, to_file=buffer, show_shapes=True, show_layer_names=True, format='png')

# Rewind the buffer's position to the beginning
buffer.seek(0)

# Open the image from the buffer using PIL
image = Image.open(buffer)

# Display the image using Streamlit
#st.image('cnn_model_dynamic.png', caption='Current CNN Model Architecture', use_column_width=True)
st.image(image, caption='Current CNN Model Architecture', use_column_width=True)

