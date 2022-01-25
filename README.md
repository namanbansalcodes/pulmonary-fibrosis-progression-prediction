# Pulmonary Fibrosis Progression Prediction

In this project, we aim to predict the FVC score for the next 50 weeks for the patient with the help of simple attributes such as age, sex, smoking status, initial FVC and the CT scan.
We also aim to build a full-stack web application where doctors can upload the patient’s details easily and in no time get the future FVC predictions for the patient.

## The Dataset

For this project, we took the dataset from the OSIC Website which had a Dataset Available 
for this. This Dataset file contains data of 175 different patients which we have used in our model.
1. ID of the patient (Used to track data of Individual Patient)
2. Week Number
3. Week Number is noted to track the patient’s lung status via FVC score check.
4. Week – 0 is when the CT scan of the patient takes place.
5. FVC - Forced vital capacity is the amount of air that can be forcibly exhaled from 
your lungs after taking the deepest breath possible, as measured by spirometry.
6. Percent – Percent is converted via FVC score which tells the status of Lung health


## Handling the CT Scan Data

Every patient has multiple DICOM files. But the attributes like the window length, window 
width, pixel spacing, etc. are the same inside each DICOM file except for the image data 
inside the DICOM files. Therefore, we first extract the statistical data from one of the DICOM files. Then, we move on to extract image data from the DICOM files.

For each image, we crop it to 512x512pixels, apply K-Means segmentation (where k=2) to segment the lung tissue from the other elements present in the CT scan such as air, bone, etc. We then use erosion and dilation which has the 
net effect of removing tiny features like pulmonary noise. Using bounding boxes for each 
image label to identify which ones represent lung and which ones represent "everything else". 
We create the masks of the segmented lung and multiply them with the original image. [10]


<img src="https://live.staticflickr.com/65535/51843452954_d97425e96f_c.jpg" width="400" height="400" alt="Lungs1">

We then merge the cropped, segmented, and normalized images from the CT scans into one 
variable and add a channel to it, this step is called dimension scaling. Now we are ready to
extract features from the images using the EfficientNet-B6 model. 

## Handling the Statistical Data
<img src="https://live.staticflickr.com/65535/51843829975_f931dcda2f_w.jpg" width="400" height="199" alt="lungs2">

## Quantile Regression
Quantile regression is a median based method that allow analysis to move away from the 
mean and see median as an alternative to least squares regression and related methods, which typically assume that the associations between independent and dependent variables are all at similar levels.

<img src="https://live.staticflickr.com/65535/51843452894_f5b2988425_w.jpg" width="400" height="117" alt="lungs3">


## Loss Function (Laplace Log)
A critical difference between probability and likelihood is in the interpretation of what is fixed and what can vary. In the case of a conditional probability, P(D|H), the hypothesis is fixed, and the data are free to vary. Likelihood, however, is the opposite. The likelihood of a hypothesis, L(H), is conditioned on the data, as if they are fixed while the hypothesis can vary. The distinction is subtle, so it is worth repeating: For conditional probability, the hypothesis is treated as a given, and the data are free to vary. For likelihood, the data are treated as a given, and the hypothesis varies.

For each true FVC measurement, you will predict both an FVC and a confidence measure. 
The metric is computed as:

<img src="https://live.staticflickr.com/65535/51843829860_f80a11abb4_w.jpg" width="400" height="153" alt="lungs4">

The error is thresholds at 1000 to avoid large errors adversely penalizing results, while the 
confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in 
FVC. The final score is calculated by averaging the metric across all the observations.[26]

## Flow Diagram of the steps
<img src="https://live.staticflickr.com/65535/51843829860_f80a11abb4_w.jpg" width="400" height="153" alt="lungs4">

## LSTM Model
<img src="https://live.staticflickr.com/65535/51843452804_2a78559e17_w.jpg" width="400" height="184" alt="lungs6">

## Training

Why let data dependencies mess with the training of the network. Therefore, we 
took K=10, and set the model to train.
For each fold, we would first calculate the base Laplace log-likelihood metric so that we can 
understand how well our model is working relative to the data. [How we do this is mentioned in 
the literature review]. Once the model would be trained, the remaining data would be used for 
validation. Even for validation, we would first calculate the base Laplace log-likelihood 
metric and compare our model’s performance.
We were aiming the validation LLL metric to be less between the range -6.0 to -6.

<img src="https://live.staticflickr.com/65535/51843201108_7a059743a4_w.jpg" width="400" height="262" alt="lungs7">

Looking at the figure it can be concluded that fold 4 did the best, as it has the lowest score. 
Remember, the more the predictions are tending over zero, the better it is.
