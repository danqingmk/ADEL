Developer Guide
===============

This guide is intended for developers and advanced users who aim to understand the internal mechanisms of ADEL or extend its functionality. It provides an overview of the code structure and detailed instructions on adding new algorithms, molecular feature types, or iterative training strategies to the framework. 
ADEL was designed to unify active learning-driven ultra-large-scale virtual screening components into a single package, integrating multiple regression models (e.g., `SVM`, `XGBoost`, `gcn`, `gat`), molecular feature extraction methods, and iterative training workflows under the hood. Understanding how these modules interact will enable you to effectively modify, optimize, or extend the software to fit specific drug discovery needs.

1. Overview of Code Structure
-----------------------------

The ADEL codebase is organized into modular components that align with its active learning-driven ultra-large-scale virtual screening workflow. 
Below is an overview of key modules, combined with core script functionalities:

    **Core Execution Module** - Acts as the entry point for model training and iterative optimization, integrating parameter parsing, multi-model parallel training, result aggregation, and iteration management (supporting configuration of models, molecular fingerprints, data splitting methods, and computing resources). This module is implemented via the ``run.py`` script.
    
    **Screening Module** - This module is used for screening data, obtaining screening results by integrating various scores and calculating indicators such as mean values：

    · ``ml_screen.py``: Responsible for virtual screening with traditional machine learning models (e.g., `SVM`, `RF`). It includes molecular fingerprint extraction, batch prediction, and score integration.
    
    · ``dl_screen.py``: Specialized in screening with deep learning models (`gcn`, `gat`). It handles molecular graph structure conversion, model loading, and high-throughput prediction for ultra-large compound libraries.
    
    **Model Algorithm Modules** - Independent scripts for each regression model: ``SVM.py``, ``RF.py``, ``XGB.py``, ``LGBM.py``, ``Ridge.py``, ``DNN.py``, ``GNN.py``. Each implements model training, hyperparameter optimization, and saving/loading logic, ensuring compatibility with the ensemble framework.
    
    **Iterative Expansion Module** - This module mainly realizes iterative expansion processing of data：
    
    · ``extract.py``: Key for training set expansion. It sorts screened molecules by score and standard deviation, filters based on score thresholds, removes duplicates with existing training data, and selects top molecules to update the training set.
    
    · ``std.py``: Calculate the mean and standard deviation (std) of the scores of various models; finally retain the columns `SMILES`, `dock_score`, `mean`` and `label`` and save them as a new CSV file.
    
    **Molecular Representation Module** - Embedded in ``ml_screen.py`` and ``run.py``, supporting molecular fingerprint generation (`ECFP4`, `MACCS`, `2d-3d`, `pubchem`) and graph structure conversion (for gcn/gat). These representations serve as input features for different model types.
    
    **Data splitting strategies** - ADEL includes multiple data splitting strategies for model validation. Likely options are:

        **random split** (shuffling the dataset into train/test)

        **scaffold split** (separating molecules by their core scaffolds so that the test set contains scaffolds not seen in training)

        **cluster-based split** (clustering molecules by similarity and then splitting clusters between train/test to ensure diversity). 

    **Utilities** - Includes result merging (file_merge function in ``run.py``), training process recording, model/parameter saving, and CSV data reading/writing, ensuring smooth workflow operation.

With this overview, you can begin to locate the areas of the code relevant to the changes you want to make. 

2. Extending ADEL
-----------------

ADEL can be extended with new algorithm models and training Set Selection Methods for Iteration. 

**Supplementing Algorithm Models**: Add a new class or function in the model module (e.g., models/ directory). The new model should follow the pattern of existing models, providing essential methods such as train (for model training) and predict (for making predictions). For third-party models (e.g., XGBoost, Random Forest), wrap their core classes (like XGBClassifier) into a new model class compatible with ADEL's interface, ensuring the required libraries are installed.Include the new model in the model registry (e.g., a configuration file or a lookup table) so that ADEL can recognize and call it by name during runtime.Validate that the model can correctly accept input features (e.g., molecular descriptors) and output predictions in the expected format (e.g., probability scores or classification labels).

**Optimizing Training Set Selection Methods for Iteration**: The current training set supplementation method in ``extract.py`` selects molecules with high uncertainty (sorted by std in descending order). Develop additional selection logic in a dedicated module (e.g., selection_strategies/). Examples include: Diversity-based selection: Select molecules with diverse structures using molecular fingerprints or Tanimoto similarity; Performance-based selection: Prioritize molecules with predicted scores within a specific range (e.g., near the decision threshold).

.. note:: To be updated