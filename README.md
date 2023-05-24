# Weather classification
The portfolio exam for Visual Analytics S22 consists of 4 projects; three class assignments and one self-assigned project. This is the repository for the fourth self assigned project.
## 1. Contributions
The code was produced by me, but with a lot of problem-solving looking into various Stack Overflow blog posts.
This youtube tutorial was especially helpful: https://www.youtube.com/watch?v=VCHNh3cMsRE
alongside this stack overflow:

## 2. Methods
Gor this project I have used the methods introduced during the Visual Analytics course to train a Convolutional Neural Network to perform weather classification.
The dataset was imported from Kaggle and is called [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
It consist of 6862 images in 11 different weather conditions (classes): dew, fogsmog, frost, glaze, hail, lightning, rain, rainbow, rime, sandstorm, snow
The data was not split into testing and training data beforehand.
The `load_data()` loads the data, preprocesses with `ImageDataGenerator()` from TensorFLow and split into testing and trining data  with the `train_test_split()` function from scikit-learn was useful for splitting the dataset.
The script produces a classification report with the ´classification_report` from scikit-learn. The classification report will located in the folder [out](https://github.com/NiGitaMyrGit/Vis_assignment_4/tree/main/out), where it predicts thelabels of the testing set.
The CNN model is built using the Sequential API from TensorFlow, consisting of multiple convolutional layers, max pooling layers, and dense layers with ReLU activation. The final layer uses softmax activation for multi-class classification. The model is furthemore compiled with the Adam optimizer, categorical entropy loss and accuracy as a metric for evaluation. 
The model is trained in the `train_model()`function, which uses with the `fit()`function from TensorFlow. 

## 3. Usage
this script was made using python 3.10.7, make sure this is your python version you run the script in.
### 3.1 clone repo
First and foremost this repository need to be cloned to your console. This can be done from the command line by running `git clone https://github.com/NiGitaMyrGit/Vis_assignment_4.git`
This will copy the repository to the location you are currently in.
### 3.2 install packages
To install packages, make sure you are located in the main directory and run the command bashe setup.sh. This will install all required packages from the `requirements.txt`file.

### 3.3 get the data
Go to the dataset [Weather Image Recognition](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) and press download. Place the download inside the folder in the main repository. This will place a zip-file called "archive.zip" inside the main directory. 
To unzip the file, run the command: `unzip archive.zip`in the command line. This will unpack a folder callede 'dataset'. This is where the data is located 
**OBS!** inside the script it's important that you change the 'dataset_path`to the aboslute path of this directory on your device. I have unfortunately not been able to make the script work with relative paths.

### 3.4 run the script
The script located in the [src](https://github.com/NiGitaMyrGit/Vis_assignment_4/tree/99dc08c20f38f2875dad2eaabc053130d141840b/src) folder contain the script.
Located in the main folder `Vis_assignment_4` run the command python3 `src/weather_CNN.py`.
The location path for the calssification report can be changed wit hthe `-r`flag eg. `python3 src/weather_CNN.py -r /path/to/report_name.txt`
### 4. discussion of results
This isn't a fancy model and it could've been optimised. It showcases that in order to get a good model, a simple script and simple training won't do. 
The accuracy is down to 0.39 with 10 epochs. For optimisation the hyperparameters could be adjusted: amount of epochs, batch size, learning rate, optimizer choice;
More convolutional layers could be added; more data augmentation like rotating, scaling, shearing, flipping; a pretrained model like VGG16 could be imported and utilized using transfer learning. 
The accuracy of the models prediction are 0.39, not overwhelmingly, but, unlike the other model I trained eg. [Assignment 3](https://github.com/NiGitaMyrGit/vis_assignment3), the script has been able to run through all the data in a fair amount of time. The biggest problem with Machine Learning, at least at this entry level working on a single computer must be the run time, which is why I chose a very simple model for this assignment. 
              precision    recall  f1-score   support

         dew       0.41      0.76      0.53       156
     fogsmog       0.43      0.60      0.50       169
       frost       0.00      0.00      0.00       103
       glaze       0.30      0.33      0.31       129
        hail       0.32      0.22      0.26       111
   lightning       0.76      0.36      0.49        78
        rain       0.39      0.48      0.43       102
     rainbow       0.23      0.21      0.22        48
        rime       0.39      0.51      0.44       212
   sandstorm       0.39      0.18      0.25       147
        snow       0.27      0.20      0.23       117

    accuracy                           0.39      1372
   macro avg       0.35      0.35      0.33      1372
weighted avg       0.36      0.39      0.35      1372
