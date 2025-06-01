.. _XReflection:

XReflection - An Easy-to-use Toolbox for Single-image Reflection Removal
====================================================================================================

XReflection is a versatile and efficient toolbox designed specifically for single-image reflection removal (SIRR). It integrates state-of-the-art solutions for training and inference, providing researchers and developers with a seamless, out-of-the-box experience for SIRR research.

---

.. _key-features:

💡 Key Features
----------------------------------------------------------------------------------------------------

- **Comprehensive Solutions** : Integrated state-of-the-art algorithms for single-image reflection removal.
- **Seamless Multi-GPU/TPU/NPU Support** : Powered by PyTorch Lightning for scalable parallelization.
- **Pretrained Model Library** : Access to a growing zoo of pretrained models for quick inference.
- **Fast Data Preparation** : High-performance synthetic data pipeline for simulation and training tasks.
- **Modular Design** : Clear and extensible architecture for customization and research.

---

.. _installation:

🚀 Installation
----------------------------------------------------------------------------------------------------

.. _requirements:

### Requirements
Make sure you have the following system dependencies installed:

- Python >= 3.8
- PyTorch >= 1.10
- PyTorchLightning >= 1.5
- CUDA >= 11.2 (for GPU support)

.. _installation-commands:

### Installation Commands

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-username/XReflection.git
   cd XReflection

   # Install dependencies
   pip install -r requirements.txt

---

.. _getting-started:

📦 Getting Started
----------------------------------------------------------------------------------------------------

.. _inference-with-pretrained-models:

### Inference with Pretrained Models
Run reflection removal on an image:

.. code-block:: python

   from xreflection import inference

   result = inference.run("path_to_your_image.jpg", model_name="default_model")

.. _training-a-model:

### Training a Model

.. code-block:: bash

   python tools/train.py --config configs/train_config.yaml

.. _data-preparation:

### Data Preparation
Generate synthetic reflection datasets:

.. code-block:: bash

   python tools/data_pipeline.py --input_dir ./raw_images --output_dir ./synthetic_data

---

.. _features-in-detail:

🌟 Features in Detail
----------------------------------------------------------------------------------------------------

.. _pretrained-model-zoo:

### Pretrained Model Zoo
Access pretrained models for various SIRR algorithms:

+---------------------+------------------------+---------------------+
| Model Name          | Description            | Performance Metrics |
+=====================+========================+=====================+
| Default Model       | General SIRR           | PSNR: 32.5, SSIM: 0.85 |
+---------------------+------------------------+---------------------+
| Enhanced Model      | Optimized structure    | PSNR: 34.3, SSIM: 0.88 |
+---------------------+------------------------+---------------------+

.. _data-pipeline:

### Data Pipeline
Create synthetic reflection datasets efficiently with custom parameters:

.. code-block:: yaml

   # Example configuration
   reflection_strength: 0.6
   blur_amount: 2.0

---

.. _development-workflow:

🛠 Development Workflow
----------------------------------------------------------------------------------------------------

.. _extending-the-toolbox:

### Extending the Toolbox
To add your custom model or feature, follow these steps:

1. Add your model implementation in ``xreflection/models``.
2. Register the model in ``xreflection/utils/model_registry.py``.
3. Update configuration.

---

.. _roadmap:

🎯 Roadmap
----------------------------------------------------------------------------------------------------

- [x] Basic training/testing pipeline
- [x] Initial pretrained models
- [ ] Support for distributed TPU training
- [ ] Enhanced dataset synthesis strategies

---

.. _citation:

📄 Citation
----------------------------------------------------------------------------------------------------

If you find XReflection useful in your research or work, please consider citing:

.. code-block:: bibtex

   @misc{xreflection2024,
     title={XReflection: A Toolbox for Single-image Reflection Removal},
     author={Your Name},
     year={2024},
     howpublished={\url{https://github.com/your-username/XReflection}}
   }

---

.. _news-and-updates:

📰 News and Updates
----------------------------------------------------------------------------------------------------

- **[2024-06-06]** Released training & testing pipelines.
- **[Upcoming]** Pretrained models for advanced architectures.

---

.. _acknowledgements:

🙏 Acknowledgements
----------------------------------------------------------------------------------------------------

Special thanks to `Google's TPU Research Cloud <https://sites.research.google/open/>`_ for providing computational resources.

---

.. _contact:

📧 Contact
----------------------------------------------------------------------------------------------------

For issues, feedback, or collaboration, reach out via GitHub Issues or email: **your-email@example.com**

---

.. _license:

📜 License
----------------------------------------------------------------------------------------------------

This project is licensed under the Apache License 2.0. See the `LICENSE <LICENSE.md>`_ file for details.