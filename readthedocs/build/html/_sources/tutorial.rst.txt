Tutorial
========

**Welcome to the ADEL tutorial**. In this tutorial, we will demonstrate a complete virtual screening workflow for drug discovery based on Active Deviation Ensemble Learning--this workflow uses molecular docking scores as the core label, and achieves efficient screening of ultra-large compound libraries through "iterative training set expansion + multi-model ensemble". 

The specific steps include: randomly selecting initial molecules from the ultra-large compound library to build a training set and obtain docking scores; after training multiple regression models, using the models to predict the scores of remaining molecules and selecting high-score, high-volatility samples to supplement the training set; repeating iterative training until the preset number of times, then selecting the optimal models to form an ensemble regression model, and finally completing virtual screening of the large-scale library.


The Virtual Screening of β₂AR Inhibitors.
------------------------------------------

The β₂-adrenergic receptor (β₂AR) is a critical drug target for treating diseases such as asthma and chronic obstructive pulmonary disease, and efficiently identifying active compounds is essential for β₂AR-related drug development.

Before diving into the code, let’s understand how to use ADEL for virtual screening. ADEL follows a three-step iterative workflow:

1. **Initial Training**: Randomly sample a small subset of molecules from the dataset to build the initial training set and train 8 heterogeneous base learners.
2. **Scoring and Selection**: Score the remaining molecules with the trained models, calculate the mean and standard deviation of the 8 model scores, and select high-scoring molecules to supplement the training set.
3. **Iterative Optimization**: Repeat the training-scoring-supplementation process for 8 iterations to build a high-performance ensemble model for large-scale virtual screening.

Prerequisites
^^^^^^^^^^^^^

Before you begin, make sure you have:

- Installed the ADEL project codebase and all dependencies
- An initial training dataset containing molecular SMILES strings and docking scores (CSV format with two core columns: SMILES and dock_score)
- A large-scale compound library for screening (CSV format, must include the SMILES field for molecular docking and model prediction)

Step 1: Prepare Your Data
^^^^^^^^^^^^^^^^^^^^^^^^^

A random selection of 1% of molecules from the ultra-large compound library is taken as the initial training set.Molecular docking simulations are performed on the molecules in the training set to obtain docking scores. These scores will serve as labeled data for subsequent model training.

For model building, you need a dataset containing molecular docking scores for regression training. The dataset should be in CSV format and include at least:

- **SMILES column** (molecular structure)
- **dock_score column** (numerical docking score of molecules)

Example training dataset:

+-----------------------------------------------+-------------------+
| SMILES                                        | dock_score        |
+===============================================+===================+
| CC(C)c1ccc(OCC(=O)NCC2(N3CCOCC3)CCCCC2)cc1    | -7.05087891609626 |
+-----------------------------------------------+-------------------+
| C[NH+]1CCN(c2ccc(CNC(=O)Nc3cccc(Cl)c3)cc2)CC1 | -8.86771625140345 |
+-----------------------------------------------+-------------------+

Step 2: Train Initial Base Regression Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Based on the molecules in the training set and their docking scores, eight regression models are constructed and trained, including Random Forest (RF), Support Vector Machine (SVM), Ridge Regression, Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LGBM), Deep Neural Network (DNN), Graph Convolutional Network (GCN), and Graph Attention Network (GAT).

Use the provided ``run.py`` script to start training by specifying the training data, model types, and number of iterations via the command line. An example command is as follows::

    python /home/models/run.py --file /home/small/train.csv --model SVM XGB RF LGBM Ridge DNN GCN GAT --iter 1

**Key Parameters Explanation**:

- ``file``: Path to training dataset
- ``model``: List of models to train (`DNN`, `SVM`, `RF`, `XGB`, `LGBM`, `Ridge`, `gcn`, `gat`)
- ``FP``: Molecular fingerprint type (`ECFP4`, `MACCS`, `2d-3d`, `pubchem`)
- ``split``: Data splitting method (`random`, `scaffold`, `cluster`)
- ``iter``: Number of training iterations, used to repeat experiments for verifying model stability
- ``threads``: Number of CPU cores for parallel computing
- ``device``: Specifies the computing device (`cpu`, `gpu`)

Step 3: Iteratively Optimize Model via Training Set Expansion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prediction of small molecule datasets using the optimal model after model training is complete. Machine learning models use ``ml_screen.py`` for prediction and deep learning models use ``dl_screen.py`` for prediction.

For machine learning models (using ``ml_screen.py``):

.. code-block:: bash

    python /home/models/ml_screen.py --file /home/database/database.csv --cpus 10 --out_dir /home/ --models /home/small/model_save/iteration_1/SVM/random_reg_ECFP4_1_SVM_bestModel.pkl

For deep learning models (using ``dl_screen.py``):

.. code-block:: python

    def screen(file='', sep=',', models=None, prop=0.5, smiles_col='Smiles', out_dir=None):
        pass

    screen(
        models ='/home/weili/zyh/dataset/keti/mukuo/fangcha/model_save/iteration_8/gcn/gcn_random_cla_0_0.03162277660168379_(128, 128)_256_5.pth',
        file='/home/weili/zyh/dataset/keti/B2AR2.csv',
        prop=0.5,sep = ',',
        out_dir='/home/weili/zyh/dataset/keti/mukuo/B2AR2/fangcha/screen_8',smiles_col='Smiles'
    )

**Key Arguments**:

- ``file``: Path to the compound library CSV
- ``models``: Directory containing the trained model
- ``out_dir``: Directory to save prediction results
- ``cpus``: Number of CPU cores to use
- ``sep``: CSV delimiter character
- ``smiles_col``: Name of the SMILES column in the library
    
The function will:

- Identify the best model based on your specifications
- Convert molecules into proper features
- Score each molecule using the model to generate a score
- Integrate the model's predicted scores, calculate the mean and standard deviation (mean and std) of the scores
- Save the results (including SMILES, individual scores, mean, and std) to a new CSV file in the specified output directory

After each round of screening, use ``extract.py`` to select high-scoring molecules and add them to the training set for the next iteration of training:

- Sort molecules by score metrics (e.g., `std` in descending order)
- Filter molecules based on score thresholds (e.g., `gcnscore` >= -15, `gatscore` >= -15)
- Remove duplicates with existing training set molecules (based on SMILES)
- Select top molecules (e.g., top 2000) to expand the training set

Step 4: Check the Results
^^^^^^^^^^^^^^^^^^^^^^^^^

After screening, the results and saved models can be found in the output directory specified by ``out_dir``. 
The output file name is derived from the input file name, with a suffix indicating the model type used for screening:

The result file includes:

- SMILES strings of the compounds
- Prediction scores generated by the model (e.g., `gcnscore` for `gcn`, `SVMscore` for `SVM`)
- Mean and standard deviation of the scores

The trained optimal models remain saved in the original model directory (specified by ``models`` during screening), ensuring reusability for subsequent experiments

Congratulations! You have now successfully completed your regression task using ADEL, predicting activity values for your compounds.