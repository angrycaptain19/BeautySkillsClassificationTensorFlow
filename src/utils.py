import os
import sys
print(sys.version, sys.platform, sys.executable)
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
from shutil import copyfile, move

def change_file_name(src_folder:str, dest_path=None,prefix=None, ext='.jpg') -> None:
    """This function rename a file and copy to the destn folder. The new file name is the src folder name. One can use any prefix other than the folder name
    using the prefix option"

    Args:
        src_folder (str): src folder path
        dest_path ([type], optional): 
        prefix ([type], optional): 
        ext (str, optional): 
    """


    try:
        print("Executing ..")
        if prefix is None:
            prefix = os.path.basename(src_folder)
        if dest_path is None:
            dest_path = src_folder        
        
        pbar = tqdm(os.listdir(src_folder))

        for id, i_file in enumerate(pbar):
            file_name = prefix + '_'+str(id) + ext
            src_file = os.path.join(src_folder, i_file)
            dstn_file = os.path.join(dest_path, file_name)
            move(src_file, dstn_file)
            pbar.set_description("Processing ..")



    except ValueError as e:
        print(e)
        exit(-1)

def validate_imgs(col='file_path'):
    df = pd.read_csv('Dataset/beauty_dataset.csv', usecols=[col])

    file_names = list(df[col])

    for filenames in file_names:
        print(filenames)
        img = tf.io.read_file(filenames)
        image_vec = tf.image.decode_jpeg(img, channels=3)

def visualize_results(fold, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        macro_f1 = history.history['macro_f1']
        val_macro_f1 = history.history['val_macro_f1']
        
        epochs = len(loss)

        style.use("bmh")
        plt.figure(figsize=(8, 8))

        plt.subplot(2, 1, 1)
        plt.plot(range(1, epochs+1), loss, label='Training Loss')
        plt.plot(range(1, epochs+1), val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.subplot(2, 1, 2)
        plt.plot(range(1, epochs+1), macro_f1, label='Training Macro F1-score')
        plt.plot(range(1, epochs+1), val_macro_f1, label='Validation Macro F1-score')
        plt.legend(loc='lower right')
        plt.ylabel('Macro F1-score')
        plt.title('Training and Validation Macro F1-score')
        plt.xlabel('epoch')

        trplot_name = './logs/train_' + str(fold) + '_plot.png'
        valplot_name = './logs/val_' + str(fold) + '_plot.png'
        plt.savefig(trplot_name)
        plt.savefig(valplot_name)
    
        return loss, val_loss, macro_f1, val_macro_f1


def perf_grid(device_, ds, target, label_names, model, n_thresh=50):
    """Computes the performance table containing target, label names,
    label frequencies, thresholds between 0 and 1, number of tp, fp, fn,
    precision, recall and f-score metrics for each label.
    
    Args:
        ds (tf.data.Datatset): contains the features array
        target (numpy array): target matrix of shape (BATCH_SIZE, N_LABELS)
        label_names (list of strings): column names in target matrix
        model (tensorflow keras model): model to use for prediction
        n_thresh (int) : number of thresholds to try
        
    Returns:
        grid (Pandas dataframe): performance table 
    """
    with tf.device(device_):
        # Get predictionsgma
        print("Getting predictions")
        y_hat_val = model.predict(ds)
        # Define target matrix
        
        target = np.array([np.asarray(lst, dtype=int) for i, lst in enumerate(target)])
        y_val = target
        # Find label frequencies in the validation set
        label_freq = target.sum(axis=0)
        # Get label indexes
        label_index = [i for i in range(len(label_names))]
        # Define thresholds
        thresholds = np.linspace(0,1,n_thresh+1).astype(np.float32)
        
        # Compute all metrics for all labels
        ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s = [], [], [], [], [], [], [], [], []
        for l in tqdm(label_index, desc="Calculating CF Matrix"):

            for thresh in thresholds:   
                ids.append(l)
                labels.append(label_names[l])
                freqs.append(round(label_freq[l]/len(y_val),2))
                y_hat = y_hat_val[:,l]
                y = y_val[:,l]
                y_pred = y_hat > thresh
                y = y.reshape(-1, 1).T
                y_pred = y_pred.reshape(-1, 1)
                tp = np.count_nonzero(y_pred  * y)
                fp = np.count_nonzero(y_pred * (1-y))
                fn = np.count_nonzero((1-y_pred) * y)
                precision = tp / (tp + fp + 1e-16)
                recall = tp / (tp + fn + 1e-16)
                f1 = 2*tp / (2*tp + fn + fp + 1e-16)
                tps.append(tp)
                fps.append(fp)
                fns.append(fn)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                
        # Create the performance dataframe
        grid = pd.DataFrame({
            'id':ids,
            'label':labels,
            'freq':freqs,
            'threshold':list(thresholds)*len(label_index),
            'tp':tps,
            'fp':fps,
            'fn':fns,
            'precision':precisions,
            'recall':recalls,
            'f1':f1s})
        
        grid = grid[['id', 'label', 'freq', 'threshold',
                    'tp', 'fn', 'fp', 'precision', 'recall', 'f1']]
        
        max_perf = grid.groupby(['id', 'label', 'freq'])[['f1']].max().sort_values('f1', ascending=False).reset_index()
        max_perf.rename(columns={'f1':'f1max_bce'}, inplace=True)
        max_perf.style.background_gradient(subset=['freq', 'f1max_bce'], cmap="Greens")
        
        return grid


if __name__ =="__main__":

    # src = "/media/sumit/Data/Workspace/SEW/dataset/blonde"
    # change_file_name(src, dest_path=None)
    # folders = [os.path.join(src, i) for i in os.listdir(src)]
    # # print(folders)
    # list(filter(lambda folds: change_file_name(folds, dest_path=folds), folders))
    # # change_file_name(
    #     src_folder="/media/sumit/Data/Workspace/SEW/dataset/balayage", dest_path=None,
    #     prefix="balayage")

    validate_imgs()