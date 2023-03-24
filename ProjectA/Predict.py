# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score
import sys


def main():
    root_path=Path('.')
    labels = ["","",""]
    args = sys.argv
    model_name = args[1]
    plot_name = args[2]
    save_path = root_path/"models"/model_name

    test_data_path=root_path/"test_dataset"/"ClassDataset_0.npz"
    

    test_data = np.load(str(test_data_path))
    test_data = test_data["arr_0"], test_data["arr_1"]

    model = keras.models.load_model(str(save_path))
    result_score = model.predict(test_data[0], verbose=0)
    
    print(f"テストデータ正解率：{accuracy_score(np.argmax(test_data[1],axis=1),np.argmax(result_score,axis=1))}")
    cm=confusion_matrix(np.argmax(test_data[1],axis=1),np.argmax(result_score,axis=1))
    plot_confusion_matrix(cm,labels,plot_name)



def plot_confusion_matrix(cm, classes,plot_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(plot_name))




if __name__ == "__main__":
    main()