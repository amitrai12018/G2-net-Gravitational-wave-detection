# G2-net-Gravitational-wave-detection
Detect gravitational wave signals from noise + signal detected binary black hole collisions, in 2015.

![image](https://user-images.githubusercontent.com/45662797/132451995-bd1b05ae-53ef-4a8a-8555-c71d189897d2.png)

*The problem-statement that is to be solved is to detect GW signals from the mergers of binary black holes. Specifically, you'll build a model to analyze simulated GW time-series data from a network of Earth-based detectors.*

## Why this is a problem in the first place?
The gravitational wave detector that were used to detect the waves have inherent noise in their output signal. So say that when the output is produced we donot clearly distinguish whether that is a noise of that contains some signal too (**Imagine the probability of (merging of black hole and of getting the signal in the detector present on the earth**).
So researchers are trying out data analysis techniques to automate the process of signal detection i.e. to find if the output signal from detector is only **'noise'** or is it **'signal+noise'**.
Following is a video on resource for reading about signal detection LIGO signal detector is explained :
[![Everything Is AWESOME](https://user-images.githubusercontent.com/45662797/132456307-31920ee2-4662-495c-916f-930003ec4406.png)](https://www.youtube.com/watch?v=B4XzLDM3Py8 "LIGO")
  
## Approach to signal detection using Deep learning Techniques.
    Basically we see this problem as binary classification problem on if the signal is detected or not.
### Dataset of various output signals 
  Link to the dataset https://www.kaggle.com/c/g2net-gravitational-wave-detection/data
  
  Data set consists of time series data containing simulated gravitational wave measurements from a network of 3 gravitational wave interferometers (LIGO Hanford, LIGO Livingston, and Virgo). Each time series contains either detector noise or detector noise plus a simulated gravitational wave signal. The task is to identify when a signal is present in the data (target=1).

The parameters that determine the exact form of a binary black hole waveform are the masses, sky location, distance, black hole spins, binary orientation angle, gravitational wave polarisation, time of arrival, and phase at coalescence (merger). These parameters (15 in total) have been randomised according to astrophysically motivated prior distributions and used to generate the simulated signals present in the data, but are not provided as part of the data-set.

Each data sample (npy file) contains 3 time series (1 for each detector) and each spans 2 sec and is sampled at 2,048 Hz.

The integrated signal-to noise ratio (SNR) is classically the most informative measure of how detectable a signal is and a typical level of detectability is when this integrated SNR exceeds ~8. This shouldn't confused with the instantaneous SNR - the factor by which the signal rises above the noise - and in nearly all cases the (unlike the first gravitational wave detection GW150914) these signals are not visible by eye in the time series.

### Data Loading and hardware requirements:
For building the model the dataset has been loaded on **Google Cloud Source** and converted to tf records so that the dataset can be easily loaded to 8 cores of kaggle **TPU '2.4.1'**
### Dataset preprocessing
The input signals are in the form of Numpy arrays. These are converted in the form of Spectrogram signals using the approach discussed [nnAudio's](https://github.com/KinWaiCheuk/nnAudio) idea of Constant Q transform its implementation is available as a library in  [CQT1992v2 layer](https://kinwaicheuk.github.io/nnAudio/_autosummary/nnAudio.Spectrogram.CQT1992v2.html?highlight=cqt1992v2#nnAudio.Spectrogram.CQT1992v2).
We convert signal from time domain to frequency domain and resize it to 256*256
**Basically the idea is to convert signal into the spectrogram signals so that the we can move from time domain to frequency domain using [Constant Q -transformation](https://en.wikipedia.org/wiki/Constant-Q_transform) so that we can detect if any particular frequency in the signal is prevelant or not. If it is then signal is present.***
Sample image of the preprocessed image is shown below:

![image](https://user-images.githubusercontent.com/45662797/132473080-4dc989e1-10b5-4b79-b744-16ae8a95f441.png)

### Model formation 
#### Efficient net Architecture B[7] is used for training using weights from ImageNet, the last layers are removed and following layers are added
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)   #adding Global AvergePooling layer for flattening
    x = tf.keras.layers.BatchNormalization()(x)       #Batch Normaization for solving  internal covariate shift if any
    x = tf.keras.layers.Dropout(0.2)(x)               #adding dropout to reduce overfitting 
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)   #output is binary so sigmoid function is applied
    model = tf.keras.Model(inputs=inputs, outputs=x)
### Model Training and validation
   * Model is trained and validated using method of 4 fold cross validation
   * For training the model is loaded on TPU and trained using Batch size 32 and 20 epochs per fold
   * The best models for each fold is saved
   * Follwoing are the 4 fold Training and validation output distribution results and accuracies 
      * Fold 1
      > Model Prediction distribution for fold 1 ↑ ![image](https://user-images.githubusercontent.com/45662797/132477413-25706cca-b766-4fca-a9df-2bee79019229.png)
      
      > Training and crossvalidation graph ↓ ![image](https://user-images.githubusercontent.com/45662797/132477616-cf6e057a-e226-43e8-83e3-f5cac5fde1ee.png)
       * Fold 2
      > Model Prediction distribution for fold 1 ↑ ![image](https://user-images.githubusercontent.com/45662797/132478210-0e321e6a-575a-456f-997c-9a5089f201ee.png)
      
      > Training and crossvalidation graph ↓![image](https://user-images.githubusercontent.com/45662797/132478252-627ca2df-474b-46d2-9831-38b258f733a8.png)
      * Fold 3
      > Model Prediction distribution for fold 3 ↑ ![image](https://user-images.githubusercontent.com/45662797/132478359-2e11c7d7-225f-47cc-bfd1-de0bd3762d86.png)
      
      > Training and crossvalidation graph ↓![image](https://user-images.githubusercontent.com/45662797/132478390-b6e57b38-dd24-44d1-8c2f-97a009bb2c6e.png)
      * Fold 4
      > Model Prediction distribution for fold 4 ↑
       ![image](https://user-images.githubusercontent.com/45662797/132478531-8b903295-84ba-43ad-a178-6078ae89390a.png)
       
      >  Training and crossvalidation graph ↓
       ![image](https://user-images.githubusercontent.com/45662797/132478572-bcd0f610-fd53-4486-b6ed-5aa45020121c.png)
   
### Model Testing 
   * For Model testing, Ensemble of these 4 models is used to predict the output, the testing accuracy is found to be 0.82
