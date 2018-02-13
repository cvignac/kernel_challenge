# Data Challenge 2018 - Kernel methods for machine learning

Raphaël Duroselle-Fourcade, Romain Girard, Clément Vignac

Description des fichiers: 
source: https://www.kaggle.com/c/kernel-methods-for-machine-learning-2017-2018/data

Principal Files
This data challenge contains three datasets. You should predict separately the labels for each dataset. For k=0, 1, 2, the main files available are the following ones

Xtrk.csv - the training sequences.
Xtek.csv - the test sequences.
Ytrk.csv - the sequence labels of the training sequences indicating bound or not.
Each Xtrk.csv (for k=0, 1, 2) contains 2000 sequences. One row represents a sequence. Similarly, each Xtek.csv contains 1000 test sequences whose Id starts from 1000k with k=0, 1, 2. These are the ones you need to predict. Once you have finished the prediction for all the datasets, you need to concatenate the results and set the corresponding Ids in order to submit a single csv file.

Ytrk.csv contains the labels corresponding to the training data, in the same format as a submission file.

Optional Files
Besides these basic data files, we also provide some additional but not necessary data for those who prefer to work directly with numeric data.

Xtrk_mat50.csv - the training feature matrices of size 2000 x 50.
Xtek_mat50.csv - the test feature matrices of size 1000 x 50.
These feature matrices are calculated respectively from Xtrk.csv and Xtek.csv based on bag of words representation. Specifically, all the subsequences of length l (here l=10) are extracted from the sequences and are represented as a vector of 4xl dimensions using one-hot encoding (with A=(1, 0, 0, 0), C=(0, 1, 0, 0), G=(0, 0, 1, 0), T=(0, 0, 0, 1)). For example, if l=3 then ACA is represented as (1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0). Then, they are clustered into 50 clusters using Kmeans and each subsequence is assigned to a label i and is represented by a binary vector whose coefficients are equal to 0 except the ith one, which is equal to 1. Finally, for each sequence, we compute the average of the representations of all its subsequences to obtain the feature vector of this sequence.
