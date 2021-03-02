import tensorflow as tf
import tensorboard
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger
import pickle
from dataset_loader import BeautyDataLoader
from model import BeautyModel
from metrics import PerformanceMetric
import datetime
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from utils import *
import config
from inferencing import Inferencing


class TrainModel:

    def __init__(self):
        
        # load model training params
        self.config,_ = config.get_config()

        # load KFold dataset
        self.dataset = []
        for kf in range(self.config.KFolds):
            self.dataset.append(BeautyDataLoader().create_dataset(fold=kf))

        # set loss function
        self.macro_f1 = PerformanceMetric().macro_F1       

        # get model architecture
        self.model = BeautyModel().create_model()

        # set tensorboard log 
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "./logs/fit/" + self.timestamp
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    def train(self):
        lr = self.config.learning_rate
        self.histories = {}
        earlystopping = EarlyStopping(monitor='val_loss', patience=10,min_delta=0.01, mode='auto')
        csv_log = CSVLogger("logs/logfiles/results.csv")
        # checkpoint = ModelCheckpoint( filepath='./ckpt/', save_freq='epoch',save_best_only=False, save_weights_only=True,verbose=1)


        self.model.compile(
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.metrics.binary_crossentropy,
              metrics=[self.macro_f1]
              )

        best_val_loss = 0.5
        for k in range(1):
            print("Processing Fold {}.... ".format(k))
            filepath='./ckpt/'+self.timestamp + '/'+str(k)+'/'

            # checkpoint = ModelCheckpoint(
            #     filepath=filepath, monitor='val_loss',
            #     save_best_only=True, save_weights_only=True,verbose=10
            #     )

            with tf.device('/gpu:0'):
                history = self.model.fit(
                    x=self.dataset[k]['train'],
                    epochs=self.config.epochs,
                    validation_data=self.dataset[k]['valid'],
                    callbacks=[self.tensorboard_callback, earlystopping, csv_log, LearningRateScheduler(self.lr_decay, verbose=1)]
                )
                print("Processing Done for {} Fold .... ".format(k))
                self.histories[k] = history
                val_loss = history.history['val_loss']
                if float(min(val_loss)) < float(best_val_loss):
                    self.model.save(filepath)

        return self.model

            
    def lr_decay(self, epoch, lr):
        if epoch != 0 and epoch % 10 == 0:
            return lr * 0.2
        return lr

    def log_perf(self, history):
        """Plot the learning curves of loss and macro f1 score 
        for the training and validation datasets.
        
        Args:
            history: history callback of fitting a tensorflow keras model 
        """
        file_name = 'logs/histories_'+self.timestamp + '.pkl'
        with open(file_name, 'wb') as handle:
            pickle.dump(self.histories, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        for k, hist in self.histories.items():
            visualize_results(fold=k, history=hist)



if __name__ == "__main__":

    trainer = TrainModel()

    model = trainer.train()
    # inference = Inferencing()
    # dataloader = BeautyDataLoader()
    # label_names = dataloader.get_label_names()
    
    # sample_test_files = [
    #     '/home/sumit/Documents/SEW/Dataset/pos/balayage/balayage_89.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/bridal_hair/bridal_121.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/bridal_hair/bridal_428.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/updo/updo_110.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/makeup/makeup_534.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/blonde/blonde_480.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/blonde/blonde_424.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/vivid/vivid_217.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/pos/haircut/haircut_228.jpg',
    #     '/home/sumit/Documents/SEW/Dataset/non_beauty/215978818.jpg'                
    #     ]
    # inference.get_predictions(sample_test_files, label_names, model)    
        # inference.get_predictions