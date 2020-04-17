# Activity-Recognition
This repository contains a trained bidirectional lstm model for Aruba dataset. The Aruba dataset is a smart home  dataset containing sensor values for 12 different activities such as sleeping,cooking. The repository also contains python code for prediction.
aruba_50_bidirectional_lstm_model.h5 : Trained model for activity recognition for window size 50.(Window size implies the set of data points taken at a time.)
final_aruba.txt : Contains the processed dataset.
change_point_detection.py : Python code to predict the activity change points in the dataset.(For each data point to detect whether it is a change point , we construct upper window and lower window of 50 data points each , pass both the windows through the trained model to obtained two vectors of size 12 each for upper&lower window. We multiply both the vectors to obtain the values(stored in other files attached) and detect the local minimas which give us the change points.)
