# csc3520-autonomous-rc-car
Finals project for machine learning. Using neural networks to predict rc car steering and throttle values.


Ensure following packages are installed properly:
* numpy
* json
* os
* CV2
* pdb
* tensorflow
* scikit-learn

How data is read in:
* read_data.py file is used in each network file to read in data from the data/ folder. It loops over the manifest.json file and then loops over the files specified in the manifest.json file to save the images and output values. It returns two arrays, the first is the array of images, the second is a 2d array where the first dimension is the throttle values and the second is the steering angle.


Then select one of the following networks to use:
* seperate_networks.py
  <br>this file uses the same data for two different networks, one which predicts throttle, the other predicts steering
* steering_to_throttle.py
  <br>this file creates two models, the first model uses data to predict steering, the second model uses data and the predicted steering to predict a throttle value
* throttle_to_steering.py
  <br>this file creates two models, the first model uses data to predict throttle, the second model uses data and the predicted throttle to predict a steering value
* img_filter_CNN.py
  <br>this file creates one model with data as input, and 2 outputs, one corresponding to throttle, and the other corresponding to steering

then run the python file
```shell
python3 <name_of_file>.py
```
The loss will be printed to the screen after the model has finished training
