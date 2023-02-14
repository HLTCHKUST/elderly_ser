import os
import argparse
import numpy as np

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from IPython.display import Audio, display




class YuemotionDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        """
        data_dir is the path to the folder with all the preprocessed .wav data
        """

        if split not in ["train", "test", "val"]:
            raise ValueError("split should be either train, test or val")

        #Build the data folders and sort the preprocessed data into male/female/elderly/non_elderly folders
        self.file_paths = self._build_data_folders(data_dir)
        print(f"Processing the {split} split...")
        train_files, test_files, val_files = self._balanced_split(self.file_paths, split_ratio=[0.5, 0.15, 0.35])
        
        #Split
        if split == "train":
            self.files = train_files
            print(f"Done, train split has {len(self.files)} samples.")
        elif split == "test":
            self.files = test_files
            print(f"Done, test split has {len(self.files)} samples.")
        elif split == "val":
            self.files = val_files
            print(f"Done, val split has {len(self.files)} samples.")

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        filename, extension = os.path.splitext(file)
        filename_data = filename.split("_")

        metadata = {"subject_id":filename_data[0],
                    "gender":("female" if filename_data[1][0]=="f" else "male"),
                    "age":filename_data[1][1:],
                    "elderly_or_not":("elderly" if int(filename_data[1][1:]) >= 59 else "non_elderly"),
                    "sentence_id":filename_data[2],
                    "format":extension}

        audio_path = os.path.join(self.data_dir, metadata["gender"], metadata["elderly_or_not"], file)

        audio, sample_rate = torchaudio.load(audio_path)
        label = filename_data[3]
        if self.transform:
            audio = self.transform(audio)

        metadata["sample_rate"] = sample_rate
        return {"audio":audio, "label":label, "metadata":metadata}



    def _balanced_split(self, file_paths, split_ratio=[0.5, 0.15, 0.35]):
        """
        split ratio default is train/val/test 0.5/0.15/0.35
        """

        train_ratio = split_ratio[0]
        val_ratio = split_ratio[1]
        test_ratio = split_ratio[2]

        train_files = []
        test_files = []
        val_files = []

        for key in file_paths.keys():
            cur_files = os.listdir(file_paths[key])
            cur_train, cur_val, cur_test = np.split(cur_files, [int(len(cur_files)*train_ratio), int(len(cur_files)*(train_ratio+val_ratio))])
            train_files.extend(cur_train.tolist())
            test_files.extend(cur_test.tolist())
            val_files.extend(cur_val.tolist())

        return train_files, test_files, val_files



    def _build_data_folders(self, data_dir):

        def _mkdir(path):
            if not os.path.isdir(path):
                os.makedirs(path)

        #Build all the relevant data folders
        male_path = os.path.join(data_dir, "male")
        _mkdir(male_path)
        female_path = os.path.join(data_dir, "female")
        _mkdir(female_path)

        male_elderly_path = os.path.join(male_path, "elderly")
        _mkdir(male_elderly_path)
        male_non_elderly_path = os.path.join(male_path, "non_elderly")
        _mkdir(male_non_elderly_path)
        female_elderly_path = os.path.join(female_path, "elderly")
        _mkdir(female_elderly_path)
        female_non_elderly_path = os.path.join(female_path, "non_elderly")
        _mkdir(female_non_elderly_path)

        file_paths = {"male_elderly_path":male_elderly_path,
                      "male_non_elderly_path":male_non_elderly_path,
                      "female_elderly_path":female_elderly_path,
                      "female_non_elderly_path":female_non_elderly_path}

        #Sort all the .wav files into relevant folders
        files = os.listdir(data_dir)
        for file in files:
            
            if "male" in file or "female" in file:
                #Already built
                continue

            metadata = file.split("_")
            age = metadata[1][1:]
            gender = metadata[1][0]
            cur_path = os.path.join(data_dir, file)

            if gender == "m":
                if int(age) >= 59:
                    new_path = os.path.join(male_elderly_path, file)
                    os.rename(cur_path, new_path)

                elif int(age) < 59:
                    new_path = os.path.join(male_non_elderly_path, file)
                    os.rename(cur_path, new_path)

            elif gender == "f":
                if int(age) >= 59:
                    new_path = os.path.join(female_elderly_path, file)
                    os.rename(cur_path, new_path)

                elif int(age) < 59:
                    new_path = os.path.join(female_non_elderly_path, file)
                    os.rename(cur_path, new_path)

            else:
                raise ValueError(f"The file {file} doesn't have the correct name formatting")
        
        file_paths = {"male_elderly_path":male_elderly_path,
                      "male_non_elderly_path":male_non_elderly_path,
                      "female_elderly_path":female_elderly_path,
                      "female_non_elderly_path":female_non_elderly_path}

        return file_paths



    def plot_waveform(self, idx):

        data = self.__getitem__(idx)
        waveform = data["audio"]
        sample_rate = data["metadata"]["sample_rate"]
        
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle("waveform")
        plt.show(block=False)



    def play_audio(self, idx):
        """
        Only works in notebook
        """
        data = self.__getitem__(idx)
        waveform = data["audio"]
        sample_rate = data["metadata"]["sample_rate"]
        
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        if num_channels == 1:
            display(Audio(waveform[0], rate=sample_rate))
        elif num_channels == 2:
            display(Audio((waveform[0], waveform[1]), rate=sample_rate))
        else:
            raise ValueError("Waveform with more than 2 channels are not supported.")


def collate_fn(batch):
    ##Need to make the masks for the padding, or not?
    #If yes, add the masks
    ## TO DO ##

    audios = [torch.squeeze(sample["audio"], dim=0) for sample in batch]
    labels = [int(sample["label"]) for sample in batch]
    longest = max([len(audio) for audio in audios])
    audios = np.stack([np.pad(audio, (0, longest - len(audio))) for audio in audios])
    audios = np.expand_dims(audios, axis=1)
    return torch.from_numpy(audios), torch.tensor(labels)

def transform(audio):
    ##If needed TO-DO
    ##Are those the usually used transformation in speech? https://pytorch.org/audio/stable/transforms.html
    ##FrequencyMasking, TimeMasking and TimeStretch?
    return audio


if __name__ == "__main__":
    ##argparse TODO

    data_dir = "preprocessed/"

    train_data = YuemotionDataset(data_dir=data_dir, split="train", transform=transform)
    test_data = YuemotionDataset(data_dir=data_dir, split="test", transform=transform)
    val_data = YuemotionDataset(data_dir=data_dir, split="val", transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

    #Test
    sample_train = next(iter(train_dataloader))
    sample_test = next(iter(test_dataloader))
    sample_val = next(iter(val_dataloader))
    print(sample_train[1])
    print(sample_test[1])
    print(sample_val[1])
