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

![Step by Step implementation to apply mask of the CT scan](https://ibb.co/1dqcsTC)

We then merge the cropped, segmented, and normalized images from the CT scans into one 
variable and add a channel to it, this step is called dimension scaling. Now we are ready to
extract features from the images using the EfficientNet-B6 model. 

## Handling the Statistical Data
![Step by Step implementation to apply mask of the CT scan](https://ibb.co/QX8zS69)



[MIT](https://choosealicense.com/licenses/mit/)
