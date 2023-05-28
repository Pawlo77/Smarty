# Smarty Project

<blockquote>
The Smarty project is an open-source end-to-end machine learning Python library. It provides both high-level and low-level APIs, enabling easy development of basic models, data preprocessing, and data visualization. Built on top of NumPy, Smarty does not support GPU operations but remains highly scalable, offering full control over data types and their structure.
</blockquote>
  
This project is an individual effort aimed at enhancing my understanding of machine learning algorithms by implementing basic functions and interfaces from scratch, using only NumPy arrays.

<hr>

For a brief description of the project:

- **Documentation**: The project includes comprehensive documentation generated by Sphinx based on comments and source Python files. To understand what Smarty is without diving into the code, simply clone the repository and open the `./docs/build/html/index.html` file.

- **Datasets Package**: This package implements basic functionality similar to Pandas DataFrames and TensorFlow Datasets, including:
  - Getting statistical information from the dataset.
  - Rearranging the dataset by adding columns, rows, renaming them, etc. (indexing by column names is allowed).
  - Configuring training properties such as batching data (with or without a remainder), setting target classes, repeating the data for multiple epochs, shuffling entries, etc.

- **Models Package**: This package contains implementations of various models, usually supporting both continuous and categorical outputs, including:
  - Dumb models.
  - Linear models.
  - Tree models.
  - Naive Bayes models.
  - K Nearest Neighbors.
  - Learning Vector Quantization.
  - All models inherit from `smarty.models.base.BaseModel`, allowing for easy customization of predictors.

- **Metrics Module**: This module provides various metrics that can be used to evaluate the performance of models.

- **Callbacks Package**: This package includes callbacks that can be used with models during batch learning, including:
  - Early Stopping.
  - Learning Rate Scheduler.

- **Preprocessing Package**: This package offers convenient data preprocessing options using basic functions, such as:
  - One-Hot Encoder.
  - Simple Encoder.
  - Standard Scaler.

- **Config Module**: This module provides configuration options for the Smarty project.

This project serves as a practical learning resource for implementing machine learning algorithms from scratch, and it offers a wide range of functionalities for data preprocessing, model building, and evaluation.

<hr>

**Note**: The project's structure and capabilities are summarized here, but for detailed information and usage, refer to the project's documentation available at `./docs/build/html/index.html`.
