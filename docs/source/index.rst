XReflection - An Easy-to-use Toolbox for Single-image Reflection Removal
======================================================================

.. image:: ./docs/logo/xreflection_logo3.png
   :alt: XReflection Logo

.. centered:: `English <README.md>`_ | `简体中文 <README_CN.md>`_

.. image:: https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue
   :target: https://img.shields.io/badge/platform-Linux%7CWindows%7CmacOS-blue
   :alt: Platform

.. image:: https://img.shields.io/pypi/pyversions/mmcv
   :target: https://img.shields.io/pypi/pyversions/mmcv
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/badge/pytorch-1.8~2.0-orange
   :target: https://pytorch.org/get-started/previous-versions/
   :alt: PyTorch

.. image:: https://img.shields.io/badge/cuda-10.1~11.8-green
   :target: https://developer.nvidia.com/cuda-downloads
   :alt: CUDA

.. image:: https://img.shields.io/pypi/v/mmcv
   :target: https://img.shields.io/pypi/v/mmcv
   :alt: PyPI

.. image:: https://img.shields.io/github/license/open-mmlab/mmcv.svg
   :target: https://img.shields.io/github/license/open-mmlab/mmcv.svg
   :alt: License

XReflection is a neat toolbox tailored for single-image reflection removal(SIRR). We offer state-of-the-art SIRR solutions for training and inference, with a high-performance data pipeline, multi-GPU/TPU/NPU support, and more!

----

.. rubric:: 📰 News and Updates

* **[Upcoming]** More models are on the way!
* **[2025-05-26]** Release a training/testing pipeline.

----

.. rubric:: 💡 Key Features

* All-in-one integration for the state-of-the-art SIRR solutions. We aim to create an out-of-the-box experience for SIRR research.
* Multi-GPU/TPU support via PyTorchLightning.
* Pretrained model zoo.
* Fast data synthesis pipeline.

----

.. rubric:: 📝 Introduction

Please visit the documentation for more features and usage.

----

.. rubric:: 🚀 Installation

.. rubric:: Requirements

Make sure you have the following system dependencies installed:

* Python >= 3.8
* PyTorch >= 1.10
* PyTorchLightning >= 1.5
* CUDA >= 11.2 (for GPU support)

.. rubric:: Installation Commands

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-username/XReflection.git
   cd XReflection

   # Install dependencies
   pip install -r requirements.txt

----

.. rubric:: 📦 Getting Started

.. rubric:: Inference with Pretrained Models

Run reflection removal on an image:

.. code-block:: python

   from xreflection import inference

   result = inference.run("path_to_your_image.jpg", model_name="default_model")

.. rubric:: Training a Model

.. code-block:: bash

   python tools/train.py --config configs/train_config.yaml

.. rubric:: Data Preparation

Generate synthetic reflection datasets:

.. code-block:: bash

   python tools/data_pipeline.py --input_dir ./raw_images --output_dir ./synthetic_data

----

.. rubric:: 🌟 Features in Detail

.. rubric:: Pretrained Model Zoo

Access pretrained models for various SIRR algorithms:

+---------------------+------------------------+---------------------+
| Model Name          | Description            | Performance Metrics |
+=====================+========================+=====================+
| Default Model       | General SIRR           | PSNR: 32.5, SSIM: 0.85 |
+---------------------+------------------------+---------------------+
| Enhanced Model      | Optimized structure    | PSNR: 34.3, SSIM: 0.88 |
+---------------------+------------------------+---------------------+

----

.. rubric:: 🙏 License and Acknowledgement

This project is licensed under the Apache License 2.0. See the `LICENSE <LICENSE.md>`_ file for details.
The authors would express gratitude to the computational resource support from Google's TPU Research Cloud.

----

.. rubric:: 📧 Contact

If you have any questions, please email **peiyuan_he@tju.edu.cn**
