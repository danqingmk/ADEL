# ADEL(Active Deviation Ensemble Learning)
AL is an active learning-based technique for ultra-large-scale virtual screening
# Install
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --torchaudio==0.12.1
```
```
pip install dgllife
```
# Example
To use ADEL, you need to set up a conda environment and install the PyTorch framework and the DGL package to support deep learning algorithms. Below is an example of using ADEL for a small dataset, provided for readers' reference.
## First step
The target proteins under study (example PDB:2RH1) were subjected to preprocessing operations using schrodinger software and docking boxes were generated. One percent of the small molecule dataset (example 17k) was randomly selected for docking to obtain the docking score.
## Second step
The algorithmic model is trained using the docking scores obtained in the previous step.
The algorithms provided in this study are SVM, XGB, RF, LGBM, Ridge, DNN, GCN, and GAT, and the reader can choose his own algorithms to use based on the examples below.
```
python /home/models/run.py --file /home/small/train.csv --model SVM XGB RF LGBM Ridge DNN GCN GAT --iter 1
```
## Third step
Prediction of small molecule datasets using the optimal model after model training is complete.Machine learning models use ml_screen.py for prediction and deep learning models use dl_screen.py for prediction.
```
python /home/models/ml_screen.py --file /home/database/databae.csv --cpus 10 --out_dir /home/ --models /home/small/model_save/iteration_1/SVM/random_reg_ECFP4_1_SVM_bestModel.pkl
```
## Forth step
The results predicted by each model were merged and the molecules were selected to expand the training set based on a selection strategy (i.e., highest prediction scores with maximum uncertainty) using std.py and extra.py.This is followed by multiple rounds of a similar iterative process.
## Last step
After several rounds of iterative process, the molecules with the highest prediction scores selected in the final round are then molecularly docked using schrodinger software to obtain potentially active molecules.
