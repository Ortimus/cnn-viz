# Dynamic CNN Model Visualizer

This Streamlit app allows users to dynamically configure and visualize Convolutional Neural Network (CNN) architectures.

## Requirements

- Anaconda or Miniconda
- See `environment.yml` for package dependencies

## Features

- Interactively adjust the number of convolutional layers
- Customize the number of filters in each convolutional layer
- Set the number of units in the dense layer
- Real-time visualization of the CNN architecture
- Display of the model summary

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/cnn-viz.git
   cd cnn-viz
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate cnnviz-env
   ```

3. Verify the installation:
   ```
   conda list
   ```

## Usage

Run the Streamlit app:

```
streamlit run src/cnn-visualize-dynamic.py
```

Use the sidebar sliders to adjust the CNN architecture:
- Number of convolutional layers
- Number of filters in each convolutional layer
- Number of units in the dense layer

The app will automatically update the model summary and architecture visualization based on your inputs.


## Updating the Environment

If you need to make changes to the environment, update the `environment.yml` file and then run:

```
conda env update -f environment.yml
```

## Dependencies

See `environment.yml` for a full list of dependencies.

## License

[MIT License](LICENSE)
