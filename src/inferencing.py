from dataset_loader import *
from utils import * 
from model import BeautyModel
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from PIL import Image

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

IMG_SIZE=224
CHANNELS=3


class Inferencing:

    def __init__(self, model_path, device_):
        with tf.device(device_): 
            self.device = device_
            # self.model = BeautyModel().create_model()
            self.model = tf.keras.models.load_model(model_path, compile=False)
            # self.model.summary()
            

        
    def create_confusion_matrix(self, model):
        dataloader = BeautyDataLoader()
        label_names = dataloader.get_label_names()
        y_val_bin = dataloader.dataset_split['valid_y']
        dataset = dataloader.create_dataset(fold=0)

        val_dataset = dataset['valid']

        target = y_val_bin[0]
        df = perf_grid(self.device, val_dataset, target, label_names, model)
        # Get the maximum F1-score for each label when using the second model and varying the threshold
        print(df.head(10))
        # return df, label_names

    def get_predictions(self, filenames, labels, model):

        from keras.preprocessing import image

        org_dataset = pd.read_csv('Dataset/beauty_dataset.csv')
        
        # Get movie info
        nrows = 5
        ncols = 2
        fig = plt.gcf()
        fig.set_size_inches(ncols * 10, nrows * 10)
        # print(filenames)
        for i, img_path in enumerate(filenames): 
            gt = org_dataset.loc[org_dataset['file_path'] == img_path, ['isbeauty', 'skill']].iloc[0]
            k,l = gt
        # Read and prepare image
            img = image.load_img(img_path, target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
            img = image.img_to_array(img)
            img = img/255
            img = np.expand_dims(img, axis=0)

            # Generate prediction
            score = model.predict(img)
            prediction = (score > 0.5).astype('int')
            prediction = pd.Series(prediction[0])
            prediction.index = labels
            prediction = prediction[prediction==1].index.values

            # Dispaly image with prediction
            # style.use('default')
            # plt.figure(figsize=(8,4))
            # plt.imshow(Image.open(img_path))
            
            # plt.show()


        
            # Set up subplot; subplot indices start at 1
            sp = plt.subplot(nrows, ncols, i + 1)
            sp.axis('Off') # Don't show axes (or gridlines)
            plt.imshow(Image.open(img_path))
            file_ = os.path.basename(img_path)
            plt.title('\n\n{}\n\GT\n{}\n\nPrediction\n{}\n'.format(file_, (k,l), list(prediction)), fontsize=9)

        plt.savefig('./logs/predictions.png')


if __name__ == "__main__":

    
    inference = Inferencing(device_='/gpu:0',model_path='./ckpt/20210226-130849/0/')
    print(inference.model.summary())
    df, labels = inference.create_confusion_matrix(inference.model)

    org_dataset = pd.read_csv('Dataset/beauty_dataset.csv')

    # print(df.head(10))

    sample_test_files = [
        '/home/sumit/Documents/SEW/Dataset/pos/balayage/balayage_89.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/bridal_hair/bridal_121.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/bridal_hair/bridal_428.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/updo/updo_110.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/makeup/makeup_534.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/blonde/blonde_480.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/blonde/blonde_424.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/vivid/vivid_217.jpg',
        '/home/sumit/Documents/SEW/Dataset/pos/haircut/haircut_228.jpg',
        '/home/sumit/Documents/SEW/Dataset/non_beauty/215978818.jpg'                
        ]
    # for i in sample_test_files:
    #     k, l = org_dataset.loc[org_dataset['file_path'] == i, ['isbeauty', 'skill']].iloc[0]
    # dataloader = BeautyDataLoader()
    # label_names = dataloader.get_label_names()
    # inference.get_predictions(sample_test_files, label_names, inference.model)

    