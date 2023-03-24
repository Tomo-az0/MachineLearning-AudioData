# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout,MaxPooling2D,Conv2D
import joblib
import tensorflow as tf
import gc
import sys


def main():
    root_path=Path('.')
    args = sys.argv
    model_name = args[1]
    save_path = root_path/"models"/model_name

    train_data_path=root_path/"train_dataset"
    
    # データが保存されているファイルパスのリスト
    data_file_paths = glob.glob(str(train_data_path/"*.npz"))
    model = define_model()

    batch_size = 8

    # ファイル単位で学習を行う
    for file_path in data_file_paths:
        

        # データを読み込む
        training_data = np.load(str(file_path))
        training_data = np.asarray(training_data["arr_0"]), np.asarray(training_data["arr_1"])

        X_train, X_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        tb_cb = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

        # モデルを現在のデータで再学習する
        model.fit(train_dataset,
                  epochs=10,
                  steps_per_epoch=None,
                  verbose=1,
                  validation_data=test_dataset,
                  callbacks=[tb_cb])

        # 現在のデータに対して、予測を行い、その精度を評価する
        accuracy = model.evaluate(X_test, y_test)
        print(f"Accuracy for file {file_path}: {accuracy}")

        # 学習したモデルを保存する
        model_file_name = file_path.replace(".npz", ".pkl")
        joblib.dump(model, model_file_name)
        del X_train, X_test, y_train, y_test,training_data,train_dataset,test_dataset
        gc.collect()
    
    print("学習完了")

    model.save(str(save_path))
    print("モデルを保存しました")



def define_model():
    model = Sequential()

    model.add(
        Conv2D(
            filters=8,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=(128, 861,1)
        ))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
        ))

    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation='relu',
        ))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten layer
    # conversion tensor of input to output-layer from 4th-order tensor to 2nd-order tensor
    model.add(Flatten())

    model.add(
        Dense(64,  # number of neuron
              activation='relu',
              kernel_initializer='random_uniform', #重みの初期値の設定
              bias_initializer='zero')) #バイアスの初期値の設定

    model.add(
        Dense(
            3,  # number of neuron (category)
            activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=tf.optimizers.Adam(),
    )

    model.summary()  # show model's summary

    return model



if __name__ == "__main__":
    main()