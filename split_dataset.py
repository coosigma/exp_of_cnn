import os, shutil
# 資料集解壓之後的目錄
original_dataset_dir = 'C:\\tmp\\train\\dogsvscats'
# 存放小資料集的目錄
base_dir = 'C:\\tmp\\dogs_and_cats_big'
os.mkdir(base_dir)
# 建立訓練集、驗證集、測試集目錄
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# 將貓狗照片按照訓練、驗證、測試分類
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
# 切割資料集
train_index = 10000
validation_index = 12500
test_index = 12500
fnames = ['cat.{}.jpg'.format(i) for i in range(train_index)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dat)
fnames = ['cat.{}.jpg'.format(i) for i in range(train_index, validation_index)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dat)
# fnames = ['cat.{}.jpg'.format(i) for i in range(validation_index, test_index)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dat = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dat)

fnames = ['dog.{}.jpg'.format(i) for i in range(train_index)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dat)
fnames = ['dog.{}.jpg'.format(i) for i in range(train_index, validation_index)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dat = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dat)
# fnames = ['dog.{}.jpg'.format(i) for i in range(validation_index, test_index)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dat = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dat)
