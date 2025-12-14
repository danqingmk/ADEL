Installation
============

This document outlines the essential requirements for successfully installing and running ADEL.
ADEL (Active Deviation Ensemble Learning) integrates multiple deep learning and machine learning models, active deviation ensemble strategies, and large-scale virtual screening optimization methods, tailored for ultra-large-scale drug discovery applications.

System Requirements
-------------------

To efficiently run ADEL, your system should meet the following specifications:

- **CPU**: Multi-core processor recommended for parallel processing tasks
- **RAM**: Minimum 16GB, 32GB or more recommended for handling large datasets
- **GPU**: CUDA-compatible GPU (optional, but recommended for deep learning models)
- **CUDA**: Version 11.3 or higher (required for GPU acceleration)
- **Disk Space**: At least 10GB for software and its dependencies

Python Environment Setup
------------------------

ADEL requires **Python 3.8**. It is strongly recommended to use **Conda** to manage your environment::

    # Create a new conda environment
    conda create -n adel_env python=3.8

    # Activate the environment
    conda activate adel_env

Core Dependency Installation
----------------------------

The following core dependencies must be installed in the specified order:

1. RDKit
^^^^^^^^

RDKit is essential for molecular structure processing and generating molecular descriptors::

    conda install rdkit -c conda-forge

2. PyTorch
^^^^^^^^^^

PyaiVS uses **PyTorch 1.12.1** for deep learning models::

    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --torchaudio==0.12.1

3. Deep Graph Library (DGL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

DGL is required for graph-based models::

    pip install dgllife

4. Additional Required Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install these extra packages required by different ADEL components::

    conda install xgboost hyperopt pandas scikit-learn numpy
    pip install requests

These dependencies support various machine learning algorithms used in the package:

+------------------------+-----------------------------------------+
| Model Type             | Required Packages                       |
+========================+=========================================+
| Machine Learning       | scikit-learn, xgboost                   |
+------------------------+-----------------------------------------+
| Deep Learning          | pytorch, dgl                            |
+------------------------+-----------------------------------------+
| Hyperparameter Opt.    | hyperopt                                |
+------------------------+-----------------------------------------+
| Data Processing        | pandas, numpy                           |
+------------------------+-----------------------------------------+

Troubleshooting
---------------

CUDA Compatibility Issues
^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter CUDA-related errors:

- Use ``nvidia-smi`` to verify your CUDA version  
- Ensure the correct CUDA version of PyTorch is installed  
- Set the appropriate environment variables::

    import os
    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

Memory Issues
^^^^^^^^^^^^^

If you run into memory errors when handling large datasets or complex models:

- Reduce batch size in model configuration  
- Use CPU mode if GPU memory is limited  
- Process datasets in chunks whenever possible

Example of specifying CPU device::
    
    python /home/models/ml_screen.py --file /home/database/databae.csv --cpus 10 --out_dir /home/ --models /home/small/model_save/iteration_1/SVM/random_reg_ECFP4_1_SVM_bestModel.pkl

Package Dependency Conflicts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you face dependency conflicts:

- Create a new Conda environment  
- Install dependencies in the exact order listed above  
- Avoid mixing conda and pip installs for the same package

Next Steps
----------

After installation, refer to the **Tutorial** for your first virtual screening task using ADEL.