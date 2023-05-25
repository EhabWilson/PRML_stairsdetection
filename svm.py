import os
from PIL import Image
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def load_data():

    # for sub_dir in os.listdir(dir):
    #     imgs = os.listdir(os.path.join(dir, sub_dir))
    #     n_samples = len(imgs)
        
    #     # labels
    #     label = int(sub_dir)
    #     labels = np.ones(n_samples) * label
    #     total_labels = np.hstack((total_labels, labels)) if label != 0 else labels

    #     # images to array
    #     imgs = [Image.open(os.path.join(dir, sub_dir, img)).convert('L') for img in imgs]
    #     imgs = [np.array(img) / 255 - 0.5 for img in imgs]    # zero mean
    #     # imgs = [np.array(img) / 255 for img in imgs]
    #     imgs = np.array(imgs).reshape(n_samples, -1)

    #     # split the data into train set and test set
    #     indexes = np.arange(n_samples)
    #     indexes = np.random.choice(indexes, size=int(np.ceil(n_samples*0.8)), replace=False)
    #     indexes_sample = np.zeros(n_samples, dtype=bool)
    #     indexes_sample[indexes] = True
    #     X_train = imgs[indexes_sample] if label == 0 \
    #         else np.concatenate((X_train, imgs[indexes_sample]))
    #     y_train = labels[indexes_sample] if label == 0 \
    #         else np.hstack((y_train, labels[indexes_sample]))
    #     X_test = imgs[~indexes_sample] if label == 0 \
    #         else np.concatenate((X_test, imgs[~indexes_sample]))
    #     y_test = labels[~indexes_sample] if label == 0 \
    #         else np.hstack((y_test, labels[~indexes_sample]))
    
    return X_train, y_train, X_test, y_test


def _svm(X_train, y_train, X_test, y_test, C=1.0, kernel='rbf', compute_confusion_matrix=False):
    t = datetime.now()
    model = svm.SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    t = datetime.now() - t

    y_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print('kernel: {}    C: {}    Train acc: {:.4g}    Test acc: {:.4g}    Training time: {:.4g}'.format(kernel, C, train_acc, test_acc, t.total_seconds()))
    if compute_confusion_matrix:
        _C = confusion_matrix(y_test, y_pred)
        for col in _C:
            print(col)
        print()


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data(os.path.join('..', 'homework4', 'data'))
    _svm(X_train, y_train, X_test, y_test, compute_confusion_matrix=True)

    # test for different kernels
    kernels = ['rbf', 'poly', 'linear', 'sigmoid']
    for kernel in kernels:
        _svm(X_train, y_train, X_test, y_test, kernel=kernel)

    # test for different C values
    C_values = [0.1, 0.5, 1, 2, 10]
    for C in C_values:
        _svm(X_train, y_train, X_test, y_test, C=C)