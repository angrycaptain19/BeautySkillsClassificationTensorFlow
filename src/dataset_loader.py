import pandas as pd
import boto3
import os
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
# import tensorflow_hub as hub


# self created modules
import config

IMG_SIZE = 224
CHANNELS = 3


class BeautyDataLoader:

    def __init__(self):
        self.config, _ = config.get_config() 
        self.df = pd.read_csv(self.config.dataset_path)        
        self.df['id'] = list(range(1, len(self.df)+1))
        # self.df = shuffle(self.df)
        self.df.set_index('id')

        # self.train_test_split()
        self.startified_splits()
        self.generate_label_encodings()
        self.transform_labels()

    def data_loader(self):
        pass
    
    def train_test_split(self, split=.10):
        pass
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)

        # self.X_train.reset_index(), self.X_test.reset_index(), self.y_train.reset_index(), self.y_test.reset_index()

    def startified_splits(self):

        self.dataset_split = {
            'train_X': [], 'train_y': [],
            'valid_X': [], 'valid_y': [],
        }
        

        skf = SKF(
            n_splits=self.config.KFolds, shuffle=True, random_state=42
            )

        data = self.df['file_path']
        labels = self.df['isbeauty']
        for train_index, test_index in skf.split(data, labels):
            
            self.dataset_split['train_X'].append([data[d] for d in train_index if d in data][:])
            self.dataset_split['valid_X'].append([data[d] for d in test_index if d in data][:])
            
            self.dataset_split['train_y'].append([
                [self.df['isbeauty'][d], self.df['skill'][d]] for d in train_index if d in labels][:])

            self.dataset_split['valid_y'].append(
               [[self.df['isbeauty'][d], self.df['skill'][d]] for d in test_index if d in labels][:])            
            
        # print(self.dataset_split)
    
    def get_label_names(self):
        return self.mlb.classes_


    def generate_label_encodings(self):
        # train_labels, valid_labels = [], []
        labels = [
            [b, sk]
            for b, sk in zip(self.df.isbeauty[1:], self.df.skill[1:])
            if b != 'isbeauty' and sk != 'skill'
        ]

        # print("Labels:")
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(labels)

        # Loop over all labels and show them
        n_labels = len(self.mlb.classes_)

        # for (i, label) in enumerate(self.mlb.classes_):
        #     print("{}. {}".format(i, label))
        
    def transform_labels(self):
        
        train_labels, valid_labels = [], []

        for train_y_set_i, valid_y_set_i in zip(self.dataset_split['train_y'], self.dataset_split['valid_y']):
            train_labels.append(list(self.mlb.transform(train_y_set_i)))
            valid_labels.append(list(self.mlb.transform(valid_y_set_i)))
        
        self.dataset_split['train_y'] = train_labels
        self.dataset_split['valid_y'] = valid_labels

    
    def normalize_img(self, filenames, label):
        """ Function to normalize image between 0 and 1

        Args:
            file_path ([str]): comple path of the image file
            label ([list]): milti-class label for the image
        """
        try:
            img = tf.io.read_file(filenames)
            image_vec = tf.image.decode_jpeg(img, channels=CHANNELS)

            # resize and normalize
            img_norm = tf.image.resize(image_vec, [IMG_SIZE, IMG_SIZE])/255.0

            return img_norm, label
        except Exception as e:
            print(e)

    
    def create_dataset(self, fold=0, is_training=True):        
        """ Here fold 0 is first dataset from all the folds created.

        Args:
            fold (int, optional): [description]. Defaults to 0.
            dataset (str, optional): [description]. Defaults to 'train'.
            is_training (bool, optional): [description]. Defaults to True.
        """
        AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically
        SHUFFLE_BUFFER_SIZE = 1024

        dataset = {}

        train_files = self.dataset_split['train_X'][fold]
        train_labels = self.dataset_split['train_y'][fold]
        valid_files = self.dataset_split['train_X'][fold]
        valid_labels = self.dataset_split['train_y'][fold]

        for typ, filenames, labels in [('train', train_files, train_labels), ('valid', valid_files, valid_labels)]:
            train_data = tf.data.Dataset.from_tensor_slices((filenames, labels))
            # print(list(train_data.as_numpy_iterator())[0])
            # normalize images        
            train_data = train_data.map(
                self.normalize_img, num_parallel_calls=AUTOTUNE
                )

            # if is_training == True:
            #     # This is a small dataset, only load it once, and keep it in memory.
            #     train_data = train_data.cache()
            #     # Shuffle the data each buffer size
            #     train_data = train_data.shuffle(buffer_size=1024)
            
            # Batch the data for multiple steps
            train_data = train_data.batch(32, drop_remainder=True)
            # Fetch batches in the background while the model is training.
            train_data = train_data.prefetch(buffer_size=self.config.autotune)

            
            dataset[typ] = train_data
            del train_data
            # print(train_data)
        print('Dataset Creation done for {} fold'.format(fold))
        return dataset

        




if __name__ == "__main__":
    obj = BeautyDataLoader()
    for i in range(5):
        print("-----------------------------------------------------------")
        tr_data = obj.create_dataset(fold=i)
        dataset = tr_data['train']
        for f, l in dataset.take(1):
            
            print("Shape of features array:", f.numpy().shape)
            print("Shape of labels array:", l.numpy().shape)