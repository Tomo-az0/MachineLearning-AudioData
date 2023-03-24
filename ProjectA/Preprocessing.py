# -*- coding: utf-8 -*-

from pathlib import Path
import glob
import librosa
import numpy as np
import scipy
from os import listdir
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
from tensorflow.keras import utils
import tensorflow as tf


def main():
    root_path=Path('.')
    labels = ["","",""]

    basis_time=5
    train_data_path=root_path/"train_dataset"
    test_data_path=root_path/"test_dataset"

    for label in labels:
        target_path=root_path/"RawData"/label

        for file_path in glob.glob(str(target_path/"*.wav")):
            file=Path(file_path).name
            print(file)
            x, sr = librosa.load(str(file_path))

            # 5秒を超えるデータは5秒となるように切り取る(切り取った際の残りの部分は5秒となるように無音部分で補う（補った際に有音部分が0.5秒(全体の10%)に満たない場合はデータとして保存しない)
            if x.shape[0]>sr*basis_time:
                x_cuts=cut_and_pad_audio(x,sr)

                for i, x_cut in enumerate (x_cuts):
                    if detect_silence(x_cut,sr)>0.1:
                        if detect_silence(x_cut,sr)>0.5:
                            # print(detect_silence(x_cut,sr))
                            save_path=root_path/"processing_data"/"cut_data"/label
                            if not save_path.is_dir():
                                save_path.mkdir(parents=True)
                            scipy.io.wavfile.write(filename=str(save_path/(str(Path(file_path).stem)+"_cut"+str(i)+".wav")), rate=sr, data=x_cut)
                        else:
                            save_path=root_path/"processing_data"/"extended_data"/label
                            if not save_path.is_dir():
                                save_path.mkdir(parents=True)
                            # print(detect_silence(x_cut,sr))
                            scipy.io.wavfile.write(filename=str(save_path/(str(Path(file_path).stem)+"_cut"+str(i)+".wav")), rate=sr, data=x_cut)
            else:
                save_path=root_path/"processing_data"/"extended_data"/label
                if not save_path.is_dir():
                    save_path.mkdir(parents=True)

                x=np.pad(x,(0, basis_time*sr - len(x)),'constant')
                scipy.io.wavfile.write(filename=str(save_path/file), rate=sr, data=x)
                audio_path=Path(file_path)
                if not (root_path/"processing_data"/"roop_data"/label).is_dir():
                    (root_path/"processing_data"/"roop_data"/label).mkdir(parents=True)
                augment_roop(audio_path,save_path=root_path/"processing_data"/"roop_data"/label)
                slide=0.5
                x_add= np.roll(x, int(slide*sr))
                scipy.io.wavfile.write(filename=str(save_path/(str(Path(file_path).stem)+"_sl"+str(slide)+".wav")), rate=sr, data=x_add)
                slide=1
                x_add= np.roll(x, int(slide*sr))
                scipy.io.wavfile.write(filename=str(save_path/(str(Path(file_path).stem)+"_sl"+str(slide)+".wav")), rate=sr, data=x_add)
                slide=1.5
                x_add= np.roll(x, int(slide*sr))
                scipy.io.wavfile.write(filename=str(save_path/(str(Path(file_path).stem)+"_sl"+str(slide)+".wav")), rate=sr, data=x_add)


    if not train_data_path.is_dir():
        train_data_path.mkdir(parents=True)
    
    target_path=root_path/"processing_data"/"extended_data"
    combine(labels,target_path,train_data_path)

    target_path=root_path/"processing_data"/"cut_data"
    combine(labels,target_path,train_data_path)

    # trainとtestデータで分ける
    for label in labels:
        target_path=train_data_path/label
        i=0
        temp=glob.glob(str(target_path/"*.wav"))
        random.shuffle(temp)
        test_num=len(temp)*0.2
        for file_path in temp:
            i+=1
            if i<test_num:
                save_path=test_data_path/label
                if not save_path.is_dir():
                    save_path.mkdir(parents=True)
                shutil.move(str(file_path), str(save_path))

    # 学習データにしか適用しない
    target_path=root_path/"processing_data"/"roop_data"
    combine(labels,target_path,train_data_path)

    for label in labels:
        target_path=train_data_path/label
        for file_path in glob.glob(str(target_path/"*.wav")):
            file=Path(file_path).name
            print(file)
            x, sr = librosa.load(str(file_path))

            x_up=change_volume(x,0.7)
            scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_v+0.7"+".wav")), rate=sr, data=x_up)
            x_up=change_volume(x,0.8)
            scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_v+0.8"+".wav")), rate=sr, data=x_up)
            x_up=change_volume(x,0.9)
            scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_v+0.9"+".wav")), rate=sr, data=x_up)
            x_up=change_volume(x,1.1)
            scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_v+1.1"+".wav")), rate=sr, data=x_up)
            x_up=change_volume(x,1.2)
            scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_v+1.2"+".wav")), rate=sr, data=x_up)

    for label in labels:
            target_path=train_data_path/label
            for file_path in glob.glob(str(target_path/"*.wav")):
                file=Path(file_path).name
                print(file)
                x, sr = librosa.load(str(file_path))
                x_noise=add_white_noise(x)
                scipy.io.wavfile.write(filename=str(target_path/(str(Path(file_path).stem)+"_+wn"+".wav")), rate=sr, data=x_noise)

    create_dataset(train_data_path,labels,spilt_num=32)
    create_dataset(test_data_path,labels,spilt_num=1)



def cut_and_pad_audio(audio_data, sample_rate):
    # 5秒に切り取るサンプル数を計算する
    five_seconds = 5 * sample_rate
    # サンプル数が5秒の倍数になるようにパディングする
    num_samples = (len(audio_data) // five_seconds + 1) * five_seconds
    padded_data = np.zeros(num_samples)
    padded_data[:len(audio_data)] = audio_data
    
    # 音響データを5秒ずつ切り取る
    num_cuts = num_samples // five_seconds
    cut_audio = np.zeros((num_cuts, five_seconds),dtype='float32')
    for i in range(num_cuts):
        cut_audio[i] = padded_data[i * five_seconds:(i+1) * five_seconds]
    
    return cut_audio


def detect_silence(x,sr, threshold=0.001):
    
    # データの絶対値がしきい値以下の箇所を検出する
    mask = np.abs(x) > threshold
    
    # マスクされた箇所のインデックスを取得する
    if(len(np.where(mask)[0])!=0):
        indices = np.where(mask)[0]
        # マスクされた箇所の先頭と末尾のインデックスを取得する
        start_idx = indices[0]
        end_idx = indices[-1]
        
        # 無音部分の割合を計算する
        duration = len(x) / sr
        silence_duration = (end_idx - start_idx) / sr
        silence_ratio = silence_duration / duration
        
        return silence_ratio
    else:
        return 0


def combine(labels, target_path, dataset_path):
    for category in labels:
        save_path = dataset_path / category
        if not (target_path / category).is_dir():
            continue
        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        for file in listdir(target_path / category):
            print(file)
            shutil.copy(str(target_path / category / file), str(save_path))


# 水増し(ループ)
def augment_roop(audio_path,save_path):
    # 5秒間の音声データを読み込む
    audio = AudioSegment.from_file(str(audio_path), format="wav", duration=5000)

    # 音がある部分だけを抽出する
    audio_without_silence = audio.strip_silence(silence_thresh=-60)

    if len(audio_without_silence)>0:

        # 出力する音声データの長さ
        output_duration = 5000

        # 音がある部分をループして、出力する音声データを作成する
        output_audio = AudioSegment.empty()
        while len(output_audio) < output_duration:
            # print(len(output_audio))
            output_audio += audio_without_silence

        # 抽出した音声データを5秒間に調整する
        five_seconds = 5 * 1000  # 5秒をミリ秒に変換する
        if len(output_audio) > five_seconds:
            # もし抽出した音声データが5秒以上の場合は、5秒間に切り詰める
            extracted_audio = output_audio[:five_seconds]
            extracted_audio.export(str(save_path/(str(audio_path.stem)+"_roop"+".wav")), format="wav")
        
        output_audio.export(str(save_path/(str(audio_path.stem)+"_roop"+".wav")), format="wav")


#水増し(音量を変更する)
def change_volume(x, rate):
    return x*rate


#水増し（ホワイトノイズ追加）
def add_white_noise(x):
    # Generate white noise
    noise = np.random.normal(scale=0.002, size=len(x))
    noise=noise.astype(np.float32)
    return x + noise


# メルスペクトグラム計算
def calculate_melsp(x,sr, n_fft=1024,n_mels = 128, hop_length=128,duration=5):
    # メルスペクトログラムの計算
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)

    # 一定の長さに調整
    n_frames = int(duration * sr / hop_length)
    if log_S.shape[1] < n_frames:
        pad_width = ((0, 0), (0, n_frames - log_S.shape[1]))
        log_S = np.pad(log_S, pad_width=pad_width, mode='constant')
    elif log_S.shape[1] > n_frames:
        log_S = log_S[:, :n_frames]
    return log_S


# 機械学習できる形にデータを加工して保存
def create_dataset(dataset_path, categories, do_classification=True,spilt_num=2):
    for j in range(spilt_num):
        dataset = []
        for class_num, category in enumerate(categories):
            
            if do_classification:
                label = class_num
            else:
                # in case of regression
                label = float(category)

            target_path=dataset_path/category
            data_list=glob.glob(str(target_path/"*.wav"))
            count=len(data_list)//spilt_num
            i=0

            print("")
            print("{}".format(category) + " -> data_size:{}".format(len(data_list)//spilt_num))

            for file_path in data_list:
                i+=1
                x, sr = librosa.load(str(file_path))
                mlsp=calculate_melsp(x,sr)
                mlsp=tf.reshape(mlsp, [mlsp.shape[0], mlsp.shape[1], 1])
                dataset.append([mlsp, label])
                
                if i>count:
                    break
            

        random.shuffle(dataset)
        X_data = []  
        y_data = []  # label

        # create data set
        for feature, label in dataset:
            X_data.append(feature)
            y_data.append(label)

        # convert to numpy array
        X_data = np.array(X_data)
        y_data = np.array(y_data)

        X_data = X_data.astype(np.float32)

        print("X_data : {}".format(X_data.shape))
        print("y_data : {}".format(y_data.shape))
        
        if do_classification:
            y_data = utils.to_categorical(y_data, len(categories))
            dataset_name = "ClassDataset"+"_"+str(j)
        else:
            dataset_name = "RegDataset"+"_"+str(j)

        save_path = dataset_path / dataset_name
        np.savez(str(save_path), X_data, y_data)



if __name__ == "__main__":
    main()