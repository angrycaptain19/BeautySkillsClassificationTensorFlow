import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import config


CHANNELS = 3
IMG_SIZE=224


class BeautyModel:
    """ 
    This classes uses tensorflow hub api to download the pretrained model. There are multiple
    models based on the type of the problem. Each model has its own input image 
    size.
    """
    def __init__(self):
        self.config, _ = config.get_config()
        url = self.config.model_url
        self.pre_trained_model_layers = hub.KerasLayer(
            url,
            input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS)
            ) 
        self.pre_trained_model_layers.trainable = False


    def create_model(self):
        """ Freeze the feature extractor layers and modifyies the classification
        layers. The top of the network is modified with the new layers corresponding 
        to our problem.
        """
        # Attaching a classification head

        self.model = tf.keras.Sequential([
            self.pre_trained_model_layers,
            layers.Dense(1024, activation='relu', name='FCN1'),
            layers.Dense(512, activation='relu', name='FCN2'),
            layers.Dense(512, activation='relu', name='FCN3'),
            layers.Dense(10, activation='sigmoid', name='Output')]
        )
        self.model.summary()

        return self.model

if __name__ == "__main__":
    obj = BeautyModel()
    obj.create_model()