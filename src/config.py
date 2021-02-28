import argparse
import tensorflow as tf

arg_lists = []
parser = argparse.ArgumentParser(description="BeautyClassifier")


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# data params
data_arg = add_argument_group("Data Params")

data_arg.add_argument(
    "--dataset_path",
    type=str,
    default="Dataset/beauty_dataset.csv",
    help="Proportion of training set used for validation",
)

data_arg.add_argument(
    "--KFolds",
    type=int,
    default=5,
    help="Proportion of training set used for validation",
)

data_arg.add_argument(
    "--valid_size",
    type=float,
    default=0.1,
    help="Proportion of training set used for validation",
)
data_arg.add_argument(
    "--batch_size", type=int, default=256, help="# of images in each batch of data"
)
data_arg.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="# of subprocesses to use for data loading",
)
data_arg.add_argument(
    "--shuffle",
    type=str2bool,
    default=False,
    help="Whether to shuffle the train and valid indices",
)

data_arg.add_argument(
    "--n_labels",
    type=int,
    default=10,
    help="Image Dimesntions",
)


data_arg.add_argument(
    "--image_dim",
    type=int,
    default=224,
    help="Image Dimesntions",
)

data_arg.add_argument(
    "--show_sample",
    type=str2bool,
    default=False,
    help="Whether to visualize a sample grid of the data",
)

data_arg.add_argument(
    "--autotune",
    type=str2bool,
    default=tf.data.experimental.AUTOTUNE,
    help="Adapt preprocessing and prefetching dynamically",
)

data_arg.add_argument(
    "--shuffle_buffer_size",
    type=str2bool,
    default=1024,
    help="To shuffle the training data by a chunck observations",
)


# training params
train_arg = add_argument_group("Training Params")
train_arg.add_argument(
    "--is_train", type=str2bool, default=True, help="Whether to train or test the model"
)

train_arg.add_argument(
    "--model_url", type=str,
    default="https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",
    help="Link to download a pre-trained model. Default is MobileNet_V2"
)
train_arg.add_argument(
    "--learning_rate", type=float, default=0.0001, help="Learning rate"
)

train_arg.add_argument(
    "--momentum", type=float, default=0.5, help="Nesterov momentum value"
)
train_arg.add_argument(
    "--epochs", type=int, default=200, help="# of epochs to train for"
)


# other params
misc_arg = add_argument_group("Misc.")
misc_arg.add_argument(
    "--use_gpu", type=str2bool, default=True, help="Whether to run on the GPU"
)
misc_arg.add_argument(
    "--best",
    type=str2bool,
    default=True,
    help="Load best model or most recent for testing",
)
misc_arg.add_argument(
    "--random_seed", type=int, default=42, help="Seed to ensure reproducibility"
)
misc_arg.add_argument(
    "--data_dir", type=str, default="./Dataset", help="Directory in which data is stored"
)
misc_arg.add_argument(
    "--ckpt_dir",
    type=str,
    default="./ckpt",
    help="Directory in which to save model checkpoints",
)
misc_arg.add_argument(
    "--logs_dir",
    type=str,
    default="./logs/",
    help="Directory in which Tensorboard logs wil be stored",
)
misc_arg.add_argument(
    "--use_tensorboard",
    type=str2bool,
    default=False,
    help="Whether to use tensorboard for visualization",
)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed