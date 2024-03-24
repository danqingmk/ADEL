import time
import os
import pandas as pd

s = time.time()

command = 'python /home/weili/zyh/models/dl_screen.py '
os.system(command=command)
dir = r'/home/weili/zyh/dataset/keti/single/14-2/guodu'
filename_csv = []
frames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        filename_csv.append(os.path.join(root, file))
        df = pd.read_csv(os.path.join(root, file), sep=',')
        frames.append(df)
print(len(filename_csv))
result = pd.concat(frames)
print(len(result))
result.to_csv('/home/weili/zyh/dataset/keti/single/14-2/screen_1/database_gat_screen.csv', index=False)
if len(os.listdir(dir)) != 0:
    for i in os.listdir(dir):
        os.remove(os.path.join(dir, i))

time.sleep(5)

command = 'python /home/weili/zyh/models/dl_screen1.py '
os.system(command=command)
dir = r'/home/weili/zyh/dataset/keti/single/14-2/guodu'
filename_csv = []
frames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        filename_csv.append(os.path.join(root, file))
        df = pd.read_csv(os.path.join(root, file), sep=',')
        frames.append(df)
print(len(filename_csv))
result = pd.concat(frames)
print(len(result))
result.to_csv('/home/weili/zyh/dataset/keti/single/14-2/screen_1/database_gcn_screen.csv', index=False)
if len(os.listdir(dir)) != 0:
    for i in os.listdir(dir):
        os.remove(os.path.join(dir, i))

time.sleep(5)

command = 'python /home/weili/zyh/models/ml_screen.py --file /home/weili/zyh/dataset/keti/data/  --cpus 21 ' \
          '--out_dir /home/weili/zyh/dataset/keti/single/14-2/guodu/  ' \
          '--models /home/weili/zyh/dataset/keti/single/14-2/model_save/iteration_1/SVM/random_reg_ECFP4_6_SVM_bestModel.pkl'
os.system(command=command)
dir = r'/home/weili/zyh/dataset/keti/single/14-2/guodu'
filename_csv = []
frames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        filename_csv.append(os.path.join(root, file))
        df = pd.read_csv(os.path.join(root, file), sep=',')
        frames.append(df)
print(len(filename_csv))
result = pd.concat(frames)
print(len(result))
result.to_csv('/home/weili/zyh/dataset/keti/single/14-2/screen_1/database_SVM_screen.csv', index=False)
if len(os.listdir(dir)) != 0:
    for i in os.listdir(dir):
        os.remove(os.path.join(dir, i))

time.sleep(5)

command = 'python /home/weili/zyh/models/ml_screen.py --file /home/weili/zyh/dataset/keti/data/  --cpus 21 ' \
          '--out_dir /home/weili/zyh/dataset/keti/single/14-2/guodu/  ' \
          '--models /home/weili/zyh/dataset/keti/single/14-2/model_save/iteration_1/XGB/random_reg_ECFP4_6_XGB_bestModel.pkl'
os.system(command=command)
dir = r'/home/weili/zyh/dataset/keti/single/14-2/guodu'
filename_csv = []
frames = []
for root, dirs, files in os.walk(dir):
    for file in files:
        filename_csv.append(os.path.join(root, file))
        df = pd.read_csv(os.path.join(root, file), sep=',')
        frames.append(df)
print(len(filename_csv))
result = pd.concat(frames)
print(len(result))
result.to_csv('/home/weili/zyh/dataset/keti/single/14-2/screen_1/database_XGB_screen.csv', index=False)
if len(os.listdir(dir)) != 0:
    for i in os.listdir(dir):
        os.remove(os.path.join(dir, i))

print(time.time() - s)
