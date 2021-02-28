# BeautySkillsClassificationTensorFlow (MultiClass-MultiLabel)

**Multi-class Multi-Label classification Using TensorFlow**

**OVERVIEW**

This project deals with the problem of classifying beauty images into beauty/non-beauty and 7 other beauty related skills (balayage, updo, makeup, haircut,
blonde, vivid, bridal hair). At the end F1 score is calculated to measure the performance of multi-class classification problem. Using metric elmenents of confusion matrix False Positive, False Negative, True Positive and True Negative are utilized to obeserve the performance of each classes.
Furthermore, Tensorboard is utilized to visualize the performance of the model in terms of valid loss, epoch loss and F1 Score.

This is an ongoing project which will be scaled from 9 classes to **70-80 classes** in future.

Keywords: Python, scikit-learn, Stratified-KFold, multi-class, AWS S3, Tensorflow, Confusion Matrix, F1 Score.
Motivation: Kaggle IMDB competition.
![alt text](https://https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex1.png?raw=true)
![alt text](https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex2.png?raw=true)
![alt text](https://github.com/Sumit1673/BeautySkillsClassificationTensorFlow/blob/main/ex3.png?raw=true)


# WorkFlow:

There are certain things are taken care before the actual process starts. The input images are transformed using JPEG encode of tensorflow, so before we proceed a function validates the dataset that it has all the valid JPEG images. Improper images are exculded from the dataset.

Step 1: Create dataset csv file.
Step 2: Validate Images using the csv file.
Step 3: Create train, valid dataset and training dataset. Encodes the categorical labels (One-hot encoding)
Step 4: For the first time it downloads the pre-trained weights of MobileNet_v2 from TensorFlow Hub.
Step 5: Initialize ModelCheckpoint, LearningRateScheduler, Early Stopping (patience=epochs). Optimizer=Adam
Step 6: Training --> Confusion Matrix , F1 score calculation --> Prediction

Logs can be observerd in tensorboard using the logs saved in 'logs' folder.

