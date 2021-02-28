# BeautySkillsClassificationTensorFlow (MultiClass-MultiLabel) on GPU

**Multi-class Multi-Label classification Using TensorFlow**

**OVERVIEW**

This project deals with the problem of classifying beauty images into beauty/non-beauty and 7 other beauty related skills (balayage, updo, makeup, haircut,
blonde, vivid, bridal hair). At the end F1 score is calculated to measure the performance of multi-class classification problem. Using metric elmenents of confusion matrix False Positive, False Negative, True Positive and True Negative are utilized to obeserve the performance of each classes.
Furthermore, Tensorboard is utilized to visualize the performance of the model in terms of valid loss, epoch loss and F1 Score.

This is an ongoing project which will be scaled from 9 classes to **70-80 classes** in future.

Keywords: Python, scikit-learn, Stratified-KFold, multi-class, AWS S3, Tensorflow, Confusion Matrix, F1 Score.
Motivation: Kaggle IMDB competition.

![] (https://https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex1.png?raw=true)

![] (https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex2.png?raw=true)

![] (https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex3.png?raw=true)

# SETUP

Requirements:
  1. Python=3.8.5
  2. TensorFlow=2.4.1
  3. Tensorboard
  4. GPU 
  5. Scikit-Learn
  6. Pandas

# NOTE:
  Some issues faced when using GPU training.
  
  If one faces GPU error related to **"Error : Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above".**
  
  Can be resolved:
    1. By freeing the GPU memory. If you are using Ubuntu try the below commands:
         
         1. check which process is utlizing the GPU and get its PID. 
    sudo fuser -v /dev/nvidia* | grep nvidia0

          2. Delete that process with PID
              sudo kill -9 <PID>
    
    2. USE THIS AT THE TOP OF THE training and prediction file to allow tensorflow know that you want the GPU  memory to 
    grow as per the requirement. If this is not initialized, then TF will hold some default memory of the GPU which is not a good way if you are short in memory.
    
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
       tf.config.experimental.set_memory_growth(physical_devices[0], True)
  

# WorkFlow:

There are certain things are taken care before the actual process starts. The input images are transformed using JPEG encode of tensorflow, so before we proceed a function validates the dataset that it has all the valid JPEG images. Improper images are exculded from the dataset.

Step 1: Create dataset csv file.

Step 2: Validate Images using the csv file.

Step 3: Create train, valid dataset and training dataset. Encodes the categorical labels (One-hot encoding)

Step 4: For the first time it downloads the pre-trained weights of MobileNet_v2 from TensorFlow Hub.

Step 5: Initialize ModelCheckpoint, LearningRateScheduler, Early Stopping (patience=epochs). Optimizer=Adam

Step 6: Training --> Confusion Matrix , F1 score calculation --> Prediction


Logs can be observerd in tensorboard using the logs saved in 'logs' folder.

