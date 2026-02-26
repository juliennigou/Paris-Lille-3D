######################################################################################
INTRODUCTION
######################################################################################

This dataset is a smaller subset of the Paris-Lille-3D Benchmark. It was subsampled with minimum distance of 0.05 m between points.


######################################################################################
TRAINING DATASET
######################################################################################

The dataset contains 7 coarse classes with:
0 unclassified
1 ground
2 buildings
3 poles
4 pedestrians
5 cars
6 vegetation

#######################################################################################
TEST DATASET
#######################################################################################

The test dataset is made of 1 of the three test clouds of Paris-Lille-3D Benchmark.

########################################################################################
SUBMISSION INSTRUCTIONS
########################################################################################

You must submit your results as a single .txt file with any filename.

The file format should be an ASCII text-file containing one class label per line. Each line must contain exactly 1 value (between 1 and 6) and corresponds to the point in the same order of the .ply point cloud file.

Classification results are evaluated automatically. You will receive an email when the evaluation of your submission has finished. This should take around 5 minutes.

The metrics used for the evaluation are based on the confusion matrix between your uploaded classification and the ground truth. We export the Intersection over Union (IoU) of each class and compute the average over the 6 classes to rank the different methods. We also provide to you the Confusion Matrix if you want to compute other metrics.

The results are first made visible only to you. You will be able to make them public at any time.


########################################################################################
SUBMISSION POLICY
########################################################################################

The test data is made to evaluate different classification methods. To allow a fair comparison with other algorithms, you are asked not to use this data and the evaluation server to tune your settings. The training data is made for that. So, you will need to wait 3h before being able to submit new results.