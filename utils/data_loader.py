import os, sys
import glob
import numpy as np
import pandas as pd
import datasets
import json
import pickle
from sklearn.model_selection import train_test_split

###
# Load Crema
###
def load_crema(dataset_path):
    # Crema
    label_dict = {
        'SAD': 'sadness', 'FEA': 'fear', 'ANG': 'angry',
        'HAP': 'happiness', 'DIS': 'disgust', 'NEU': 'neutral'
    }
    data_dict = {'audio': [], 'age': [], 'gender': [], 'ethnic': [], 'lang': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    meta_df = pd.read_csv(f'{dataset_path}/Crema/VideoDemographics.csv')
    for path in glob.glob(f'{dataset_path}/Crema/AudioWAV/*'):
        actor_id, uttr_id, label = path.split('/')[-1].split('_')[0:3]
        age = meta_df.loc[meta_df['ActorID'] == int(actor_id), 'Age'].values[0]
        gender = meta_df.loc[meta_df['ActorID'] == int(actor_id), 'Sex'].values[0]
        ethnic = meta_df.loc[meta_df['ActorID'] == int(actor_id), 'Race'].values[0]
        label = label_dict[label]

        data_dict['audio'].append(path)
        for emotion in label_dict.values():
            if label == emotion:
                data_dict[emotion].append(1)
            else:
                data_dict[emotion].append(0)
        data_dict['age'].append(age)
        data_dict['gender'].append(gender.lower())
        data_dict['ethnic'].append(ethnic.lower())
        data_dict['lang'].append('english')

    crema_df = pd.DataFrame(data_dict)
    return crema_df

###
# Load ElderReact
###
def load_elder_react(dataset_path):
    data_columns = ['audio', 'angry', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'gender', 'valence']
    elder_train_df = pd.read_csv(f'{dataset_path}/ElderReact/ElderReact-master/Annotations/train_labels.txt', sep=' ', header=None)
    elder_valid_df = pd.read_csv(f'{dataset_path}/ElderReact/ElderReact-master/Annotations/dev_labels.txt', sep=' ', header=None)
    elder_test_df = pd.read_csv(f'{dataset_path}/ElderReact/ElderReact-master/Annotations/test_labels.txt', sep=' ', header=None)

    elder_train_df.columns = data_columns
    elder_valid_df.columns = data_columns
    elder_test_df.columns = data_columns

    elder_train_df['lang'] = 'english'
    elder_valid_df['lang'] = 'english'
    elder_test_df['lang'] = 'english'

    elder_train_df['audio'] = elder_train_df['audio'].apply(lambda x: f'{dataset_path}/ElderReact/ElderReact_train/{x}'.replace('.mp4','.wav'))
    elder_valid_df['audio'] = elder_valid_df['audio'].apply(lambda x: f'{dataset_path}/ElderReact/ElderReact_train/{x}'.replace('.mp4','.wav'))
    elder_test_df['audio'] = elder_test_df['audio'].apply(lambda x: f'{dataset_path}/ElderReact/ElderReact_train/{x}'.replace('.mp4','.wav'))

    elder_train_df['gender'] = elder_train_df['gender'].apply(lambda x: 'male' if x == 'm' else 'female')
    elder_valid_df['gender'] = elder_valid_df['gender'].apply(lambda x: 'male' if x == 'm' else 'female')
    elder_test_df['gender'] = elder_test_df['gender'].apply(lambda x: 'male' if x == 'm' else 'female')

    elder_train_df['split'] = 'train'
    elder_valid_df['split'] = 'valid'
    elder_test_df['split'] = 'test'

    elder_train_df['age'] = 65
    elder_valid_df['age'] = 65
    elder_test_df['age'] = 65

    elder_df = pd.concat([elder_train_df, elder_valid_df, elder_test_df])
    return elder_df

###
# Load ESD
###
speaker_id_map = {
    '0001': ('Chinese', 'Female'),
    '0002': ('Chinese', 'Female'),
    '0003': ('Chinese', 'Female'),
    '0004': ('Chinese', 'Male'),
    '0005': ('Chinese', 'Male'),
    '0006': ('Chinese', 'Male'),
    '0007': ('Chinese', 'Female'),
    '0008': ('Chinese', 'Male'),
    '0009': ('Chinese', 'Female'),
    '0010': ('Chinese', 'Male'),
    '0011': ('English', 'Male'),
    '0012': ('English', 'Male'),
    '0013': ('English', 'Male'),
    '0014': ('English', 'Male'),
    '0015': ('English', 'Female'),
    '0016': ('English', 'Female'),
    '0017': ('English', 'Female'),
    '0018': ('English', 'Female'),
    '0019': ('English', 'Female'),
    '0020': ('English', 'Male')
}

def load_esd(dataset_path):
    label_dict = {'Angry': 'angry', 'Happy': 'happiness', 'Neutral': 'neutral', 'Sad': 'sadness', 'Surprise': 'surprise'}
    data_dict = {'audio': [], 'gender': [], 'lang': [], 'split': [], 'age': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    for folder_path in glob.glob(f'{dataset_path}/ESD/*[!.txt]'):
        lang, gender = speaker_id_map[folder_path.split('/')[-1]]
        for emot_k, emot_v in label_dict.items():
            for split_path in glob.glob(f'{folder_path}/{emot_k}/*'):
                split = split_path.split('/')[-1]
                for path in glob.glob(f'{split_path}/*.wav'):
                    data_dict['audio'].append(path)
                    for emotion in label_dict.values():
                        if emot_v == emotion:
                            data_dict[emotion].append(1)
                        else:
                            data_dict[emotion].append(0)
                    data_dict['lang'].append(lang.lower())
                    data_dict['gender'].append(gender.lower())
                    data_dict['split'].append(split.lower())
                    data_dict['age'].append(30)

    esd_df = pd.DataFrame(data_dict)
    return esd_df

###
# Load CSED
###
def load_csed(dataset_path):
    label_dict = {'pos': 'positive', 'nat': 'neutral', 'neg': 'negative'}
    data_dict = {'audio': [], 'gender': [], 'lang': [], 'age': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    for path in glob.glob(f'{dataset_path}/Chinese-Speech-Emotion-Datasets/Training Data (wav)/*/*.wav'):
        meta = path.split('/')[-1].lower().replace('_','-')
        emot_meta, gender = meta.split('-')[:2]
        label = label_dict[emot_meta]

        data_dict['audio'].append(path)
        for emotion in label_dict.values():
            if label == emotion:
                data_dict[emotion].append(1)
            else:
                data_dict[emotion].append(0)
        data_dict['gender'].append(gender)
        data_dict['lang'].append('chinese')
        data_dict['age'].append(65)

    csed_df = pd.DataFrame(data_dict)
    return csed_df

###
# Load TESS
###
def load_tess(dataset_path):
    label_dict = {
        'angry': 'angry', 'disgust': 'disgust', 'fear': 'fear', 'ps': 'surprise',
        'happy': 'happiness', 'neutral': 'neutral', 'sad': 'sadness'
    }
    data_dict = {'audio': [], 'gender': [], 'age': [], 'lang': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    for path in glob.glob(f'{dataset_path}/TESS/TESS Toronto emotional speech set data/*/*.wav'):
        meta = path.split('/')[-1].split('.')[0].lower()
        age_meta, sent_meta, label_meta = meta.split('_')
        age = 64 if age_meta == 'oaf' else 26
        label = label_dict[label_meta]

        data_dict['audio'].append(path)
        for emotion in label_dict.values():
            if label == emotion:
                data_dict[emotion].append(1)
            else:
                data_dict[emotion].append(0)
        data_dict['age'].append(age)
        data_dict['gender'].append('female')
        data_dict['lang'].append('english')

    tess_df = pd.DataFrame(data_dict)
    return tess_df

###
# Load IEMOCAP
###
def load_iemocap(dataset_path):
    label_dict = {
        'ang': 'angry', 'dis': 'disgust', 'exc': 'excitement', 'fea': 'fear', 
        'fru': 'frustrated', 'hap': 'happiness', 'neu': 'neutral', 'oth': 'other',
        'sad': 'sadness', 'sur': 'surprise', 'xxx': 'unknown'
    }
    data_dict = data_dict = {'audio': [], 'gender': [], 'age': [], 'lang': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    metadata = pickle.load(open(f'{dataset_path}/IEMOCAP/meta.pkl', 'rb'))
    for path in glob.glob(f'{dataset_path}/IEMOCAP/Ses*/*/audio.wav'):
        session_id = path.split('/')[-2]
        gender = 'male' if session_id.split('_')[0][-1] == 'M' else 'female'
        label = label_dict[metadata[session_id]['label']]

        data_dict['audio'].append(path)
        for emotion in label_dict.values():
            if label == emotion:
                data_dict[emotion].append(1)
            else:
                data_dict[emotion].append(0)
        data_dict['gender'].append(gender)
        data_dict['age'].append(30)
        data_dict['lang'].append('english')
        
    iemocap_df = pd.DataFrame(data_dict)
    return iemocap_df

###
# Load CMU-MOSEI
###
def load_cmu_mosei(dataset_path):
    label_dict = {
        0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happiness',  4: 'sadness', 5: 'surprise'
    }
    data_dict = data_dict = {'audio': [], 'lang': []}
    data_dict.update({ emotion: [] for emotion in label_dict.values()})

    metadata = pickle.load(open(f'{dataset_path}/CMU_MOSEI/meta.pkl', 'rb'))
    for path in glob.glob(f'{dataset_path}/CMU_MOSEI/*/audio.wav'):
        session_id = path.split('/')[-2]
        labels = metadata[session_id]['label']

        data_dict['audio'].append(path)
        data_dict['lang'].append('english')
        for i, label in enumerate(labels):
            data_dict[label_dict[i]].append(label)
            
    cmu_mosei_df = pd.DataFrame(data_dict)
    cmu_mosei_df['age'] = -100
    return cmu_mosei_df

###
# Split dataset into train, valid, & test
###
def assign_dataset_split(df, val_size, test_size, stratify_col='gender', seed=0):
    tr_val_df, test_df = train_test_split(df, stratify=df[stratify_col], test_size=test_size, random_state=seed)
    train_df, valid_df = train_test_split(tr_val_df, stratify=tr_val_df[stratify_col], test_size=val_size, random_state=seed)
    train_df.loc[:,'split'] = 'train'
    valid_df.loc[:,'split'] = 'valid'
    test_df.loc[:,'split'] = 'test'
    
    return pd.concat([train_df, valid_df, test_df])

###
# Function to retrieve and aggregate datasets
###
def retrieve_aggregate_datasets(dataset_path):
    # Load datasets
    crema_df = load_crema(dataset_path)
    elder_df = load_elder_react(dataset_path)
    esd_df = load_esd(dataset_path)
    csed_df = load_csed(dataset_path)
    tess_df = load_tess(dataset_path)
    iemocap_df = load_iemocap(dataset_path)
    cmu_mosei_df = load_cmu_mosei(dataset_path)
    
    # Add dataset meta
    crema_df['dataset'] = 'crema'
    elder_df['dataset'] = 'elder_react'
    esd_df['dataset'] = 'esd'
    csed_df['dataset'] = 'csed'
    tess_df['dataset'] = 'tess'
    iemocap_df['dataset'] = 'iemocap'
    cmu_mosei_df['dataset'] = 'cmu_mosei'

    # Add age group
    elderly_threshold = 60
    def get_age_group(age):
        if age == -100:
            return 'unknown'
        elif age >= elderly_threshold:
            return 'elderly'  
        else: 
            return 'others'

    crema_df['age_group'] = crema_df['age'].apply(get_age_group)
    elder_df['age_group'] = elder_df['age'].apply(get_age_group)
    esd_df['age_group'] = esd_df['age'].apply(get_age_group)
    csed_df['age_group'] = csed_df['age'].apply(get_age_group)
    tess_df['age_group'] = tess_df['age'].apply(get_age_group)
    iemocap_df['age_group'] = iemocap_df['age'].apply(get_age_group)
    cmu_mosei_df['age_group'] = cmu_mosei_df['age'].apply(get_age_group)

    # Split Elderly and Others
    crema_elderly_df = crema_df[crema_df['age_group'] == 'elderly']
    crema_others_df = crema_df[crema_df['age_group'] == 'others']
    tess_elderly_df = tess_df[tess_df['age_group'] == 'elderly']
    tess_others_df = tess_df[tess_df['age_group'] == 'others']
    
    # Assign Split
    csed_df = assign_dataset_split(csed_df, val_size=52, test_size=400)
    crema_elderly_df = assign_dataset_split(crema_elderly_df, val_size=42, test_size=300)
    crema_others_df = assign_dataset_split(crema_others_df, val_size=750, test_size=1200)
    tess_elderly_df = assign_dataset_split(tess_elderly_df, val_size=200, test_size=500)
    tess_others_df = assign_dataset_split(tess_others_df, val_size=201, test_size=500)
    iemocap_df = assign_dataset_split(iemocap_df, val_size=1039, test_size=1500)
    cmu_mosei_df = assign_dataset_split(cmu_mosei_df, val_size=1259, test_size=2000, stratify_col='age_group')

    # Combined DataFrame
    combined_df = pd.concat([
        crema_others_df, crema_elderly_df, elder_df, esd_df, csed_df, 
        tess_others_df, tess_elderly_df, iemocap_df, cmu_mosei_df
    ])
    
    # Preprocess empty value
    numeric_columns = [
        'sadness', 'fear', 'angry', 'happiness', 'disgust', 'neutral',  
        'happiness', 'surprise', 'valence', 'positive', 'negative',
        'excitement', 'frustrated', 'other', 'unknown', 'age'
    ]

    for column in combined_df.columns:
        if column in numeric_columns:
            combined_df[[column]] = combined_df[[column]].fillna(-100)
        else:
            combined_df[[column]] = combined_df[[column]].fillna('unknown')

    # Return the merged DataFrame
    return combined_df

###
# Convert DataFrame to HF Dataset
###
def df_to_dataset(df):
    return datasets.Dataset.from_pandas(df).cast_column("audio", datasets.features.Audio(sampling_rate=16000))

###
# Main Function for Dataset Loading
###
def load_dataset(dataset_path):
    label_list = [
        'sadness', 'fear', 'angry', 'happiness', 'disgust', 'neutral', 'surprise', 
        'positive', 'negative', 'excitement', 'frustrated', 'other', 'unknown'
    ]
    
    combined_df = retrieve_a ggregate_datasets(dataset_path)    
    combined_df['labels'] = combined_df.apply(lambda row: [int(row[label]) for label in label_list], axis=1)
    combined_df = combined_df[list(set(list(combined_df.columns)) - set(label_list + ['valence']))]

    en_others_df = combined_df.loc[(combined_df['lang'] == 'english') & (combined_df['age_group'] == 'others')]
    en_elderly_df = combined_df.loc[(combined_df['lang'] == 'english') & (combined_df['age_group'] == 'elderly')]
    zh_others_df = combined_df.loc[(combined_df['lang'] == 'chinese') & (combined_df['age_group'] == 'others')]
    zh_elderly_df = combined_df.loc[(combined_df['lang'] == 'chinese') & (combined_df['age_group'] == 'elderly')]
    
    trn_en_others_df = en_others_df.loc[en_others_df['split'] == 'train']
    val_en_others_df = en_others_df.loc[en_others_df['split'] == 'valid']
    tst_en_others_df = en_others_df.loc[en_others_df['split'] == 'test']
    
    trn_en_elderly_df = en_elderly_df.loc[en_elderly_df['split'] == 'train']
    val_en_elderly_df = en_elderly_df.loc[en_elderly_df['split'] == 'valid']
    tst_en_elderly_df = en_elderly_df.loc[en_elderly_df['split'] == 'test']
    
    trn_zh_others_df = zh_others_df.loc[zh_others_df['split'] == 'train']
    val_zh_others_df = zh_others_df.loc[zh_others_df['split'] == 'valid']
    tst_zh_others_df = zh_others_df.loc[zh_others_df['split'] == 'test']
    
    trn_zh_elderly_df = zh_elderly_df.loc[zh_elderly_df['split'] == 'train']
    val_zh_elderly_df = zh_elderly_df.loc[zh_elderly_df['split'] == 'valid']
    tst_zh_elderly_df = zh_elderly_df.loc[zh_elderly_df['split'] == 'test']
    
    return [
        {
            "lang": "eng",
            "group": "others",
            "data": (
                df_to_dataset(trn_en_others_df),
                df_to_dataset(val_en_others_df),
                {dset: df_to_dataset(df) for dset, df in tst_en_others_df.groupby('dataset')}
            )
        }, {
            "lang": "eng",
            "group": "elderly",
            "data": (
                df_to_dataset(trn_en_elderly_df),
                df_to_dataset(val_en_elderly_df),
                {dset: df_to_dataset(df) for dset, df in tst_en_elderly_df.groupby('dataset')}
            )
        }, {
            "lang": "zho",
            "group": "others",
            "data": (
                df_to_dataset(trn_zh_others_df),
                df_to_dataset(val_zh_others_df),
                {dset: df_to_dataset(df) for dset, df in tst_zh_others_df.groupby('dataset')}
            )
        }, {
            "lang": "zho",
            "group": "elderly",
            "data": (
                df_to_dataset(trn_zh_elderly_df),
                df_to_dataset(val_zh_elderly_df),
                {dset: df_to_dataset(df) for dset, df in tst_zh_elderly_df.groupby('dataset')}
            )
        },
    ]
    