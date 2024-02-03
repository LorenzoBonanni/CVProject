# Computer Vision Project: Medical Image Segmentation

## Overview

This repository contains the code and resources for our computer vision project focused on medical image segmentation. Our goal is to develop accurate models that can identify and segment objects within medical images.
In this report, we perform critical evaluation and comparison of two distinct methodologies for medical image segmentation:

1. **UNet**:
    - UNet is a convolutional neural network architecture.
    - It has demonstrated robust performance in semantic segmentation tasks, particularly in medical imaging.
    - UNet effectively captures both local and global features.

2. **MAE-UNETR Hybrid**:
    - The MAE-UNETR hybrid combines the strengths of masked autoencoders (MAE) and the transformer-based UNETR model.
    - MAE contributes to feature extraction, while UNETR leverages attention mechanisms and hierarchical representations.
    - This hybrid approach offers a potentially enhanced segmentation capability.

## Installation and Setup

1. **Python Version**: This project requires **Python 3.10**. Make sure you have it installed.

2. **Dependencies**: Install the necessary libraries by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

## Datasets

We used the following datasets for training and evaluation:

- **CVC ClinicDB**:
    - [CVC ClinicDB on Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb)
    - Contains endoscopic images for polyp detection.

- **Brain Tumor Segmentation (BTS)**:
    - [Brain Tumor Segmentation on Kaggle](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)
    - MRI scans with annotated tumor regions.

- **Breast Ultrasound Images (BUSI)**:
    - [Breast Ultrasound Images Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
    - Ultrasound images for breast lesion detection.

## Reproducing Results

To reproduce the results from our report, follow these steps:
1. **Execution**
    - **Manual Execution**:
      - Open `experiments.txt`.
      - Adjust the dataset directory paths according to your local setup.
      - Run the commands listed inside the file.
    
    - **Automated Execution**:
      - Use the `run.py` script.
      - It will sequentially execute all experiments listed in `experiments.txt`.
      - Ensure that the dataset paths are correctly configured in the script.

2. **Pretrained Models**:
    - Create a `pretrained` directory to store the best models obtained during training.

## Contributing

Contributions are welcome! If you find any issues or have improvements, feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
