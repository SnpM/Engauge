
# Engauge
Engauge is the engagement level tracker application developed for John Hopkins University Applied Physics Laboratory. Our project explored potential uses engagement tracking metrics in the contexts of DOD, project management, and education. Engauge demonstrates the use of facial recognition technologies to predict engagement levels in subjects looking at the screen.

## Requirements
- OS: Ubuntu 20.04.1 LTS
- Specs: [D2as_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/dav4-dasv4-series) on Azure

## Setup
Install Python3.6.
```
sudo add-apt-repository ppa:deadsnakes/ppa
apt-get update
sudo apt-get install python3.6
```

Install Python dependencies.
```
python3.6 -m pip install tensorflow==1.5.5 matplotlib pandas
```

Install OpenFace. OpenFace is an open-source facial recognition project that collects features for our model predictions. Run the following code from outside the Engauge folder.
```
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash ./download_models.sh
sudo bash ./install.sh
```

Rebuild OpenFace with the modified files. The following shell code assumes that the OpenFace and Engauge projects have the same parent folder.
```
cp Engauge/OpenFaceMods/FeatureExtraction.cpp OpenFace/exe/FeatureExtraction/FeatureExtraction.cpp
cp Engauge/OpenFaceMods/SequenceCapture.cpp OpenFace/lib/local/Utilities/src/RecorderCSV.cpp
cp Engauge/OpenFaceMods/RecorderCSV.cpp OpenFace/lib/local/Utilities/src/SequenceCapture.cpp
cp Engauge/OpenFaceMods/SequenceCapture.h lib/local/Utilities/include/SequenceCapture.h
```

Run the program.
```
cd Enguage
./predict.py
```

## Known Issues
The model produces several Numpy warnings and anomalous results produced that did not significantly impact the demonstration, but should be fixed for production. Our hardware also didn't have a GPU, so low frame rate may have affected the prediction performance. For future iterations of this project, I recommend porting the code to the most recent version of Python and TensorFlow and taking advantage of a GPU for more feature frames per second and lower latency calculations.

## Credits
This demonstration was developed by John Pan with the ideas and help of my teammates: Caleb Song, Weiss Yuan, Soleil Xie, Alex Gannon, and Kyle Yelensky. Our APL Circuit mentors--Aishwarya Jayabharathi, Jorge Rivera, and Will Gray-Roncal--guided our research and demonstrations. This application uses the models and scripts in the [Engagement-Level-Reader](https://github.com/AnshulSood11/Engagement-Level-Prediction) project. We modified their scripts for demonstration purposes.

The following section contains the README for the original Engagement-Level-Prediction project.

-----------------------------------------------

# Engagement-Level-Prediction
Real-time engagement-intensity prediction.

## Functioning
The project uses the award-winning approach proposed by the winners [1] of the engagement prediction task, a sub-challenge of the Emotion Recognition in the Wild Challenge (EmotiW 2019).

![pipeline](https://github.com/AnshulSood11/Teaching-Quality-Evaluation-Using-Engagement-Intensity-Prediction/blob/master/engagement-intensity-images/Pipeline.jpg)

### 1. Data Acquisition:

Video input is taken using the webcam mounted on the device. This input video stream is divided into ten seconds segments for evaluation. The divided segments are further sent in the order further down the model.

### 2. Data Preprocessing:

![facial-features](https://github.com/AnshulSood11/Teaching-Quality-Evaluation-Using-Engagement-Intensity-Prediction/blob/master/engagement-intensity-images/Screenshot%20from%202019-11-29%2012-30-14.png)

To obtain facial features, the video is fed into OpenFace. OpenFace gives 300 feature-points about eye-gaze-angles, eye-direction-vectors, 2-D eye-landmarks, 3-D eye-landmarks, and head pose. This is done for each video segment which consists of 100 frames. Further, the video is divided into 15 segments, for each segment, the standard deviation and mean is calculated for all the features to calculate the changes in values across the frames, this gives us about 60 feature-points. Finally, after the preprocessing is done, we get a 15x60 feature-set for each 10-second video segment.

### 3. Cognitive Engagement-Intensity Regression:

The feature-set is then fed into a deep neural network. The architecture of the neural network consists of two LSTM layers, two fully-connected layers, and a global average pooling layer. The model then predicts the cognitive engagement intensity as a value between 0 and 1.
Two different models were trained with slightly different hyperaprameters. Average of the values predicted by each of the two is taken.

### 4. Categorical Classification:

The engagement-level is classified into four levels: Disengaged, Barely-engaged, Engaged and Highly-engaged. The value range for each level is as follows:

0 <= engagement-intensity < 0.4 : Disengaged

0.4 <= engagement-intensity < 0.6: Barely-Engaged

0.6 <= engagement-intensity < 0.83: Engaged

0.83 <= engagement-intensity <=1.00 : Highly-engaged

### 5. Data-Presentation:

![data-presentation](https://github.com/AnshulSood11/Teaching-Quality-Evaluation-Using-Engagement-Intensity-Prediction/blob/master/engagement-intensity-images/Figure_1.png)

## Requirements

1. Linux
2. Nvidia GPU

## Geting Started

1. Install [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)

2. Make the following modifications in OpenFace:

    a. Replace the FeatureExtraction.cpp in OpenFace/exe/FeatureExtraction with FeatureExtraction.cpp in OpenFace mods folder.

    b. Do the same for SequenceCapture.cpp in OpenFace/lib/local/Utilities/src and for SequenceCapture.h in OpenFace/lib/local/Utilities/include

    c. Go to OpenFace/build and exectute ```make```.

3. Delete the folder OpenFace/build/processed

4. Open terminal in OpenFace/build directory and run:
```bash
./bin/FeatureExtraction -wild -device 0 -pose -gaze -2Dfp -3Dfp
```
  This starts the video input and starts storing preprocessed data in the OpenFace/build/processed directory.

5. From another terminal in the repository folder, run:
```bash
python predict.py
```

## Dataset

The model was trained on a closed dataset “Engagement in the wild dataset” [2]. The data was recorded with a webcam on a laptop or computer, a mobile phone camera while the student participants were watching five minutes long MOOC video. The environment in which students watched the course videos varied from the computer lab, canteen, playground to hostel rooms. The dataset includes 91 subjects (27 females and 64 males) with 147, 48 and 67 videos, each approximately 5 minutes long, for training, validation, and testing, respectively. Four levels of engagement {0,1,2,3} were labeled by annotators. In this task, the problem was formulated as a regression problem with the output in the range \[0,1] corresponding to 4 engagement levels (0 : 0, 0.33: 1, 0.66: 2, 1 : 3). The system performance was evaluated with mean square error (MSE) between the ground truth, and the predicted value of the test set.

## References

[1] Van Thong Huynh, Hyung-Jeong Yang, Guee-Sang Lee, and Soo-
Hyung Kim. 2019. Engagement Intensity Prediction with Facial Be-
havior Features. In 2019 International Conference on Multimodal In-
teraction (ICMI ’19), October 14–18, 2019, Suzhou, China. ACM, New
York, NY, USA, 5 pages. https://doi.org/10.1145/3340555.3355714

[2] Amanjot Kaur, Aamir Mustafa, Love Mehta, and Abhinav Dhall. 2018. Prediction and localization of student engagement in the wild. In 2018 Digital Image Computing: Techniques and Applications (DICTA). IEEE, 1–8.

## Acknowledgements

* [th2l/SML_EW_EmotiW2019](https://github.com/th2l/SML_EW_EmotiW2019) Update: This repository has been taken down by its [owner](https://github.com/th2l).
