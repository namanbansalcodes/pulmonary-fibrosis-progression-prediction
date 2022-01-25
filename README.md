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

## Conclusion & Future Work
Now that both the aims of the model have been achieved, we can compare our validation accuracy to 
that of other solution that we discussed in the literature review.
EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales 
all dimensions of depth/width/resolution using a compound coefficient.
In general, the EfficientNet models achieve both higher accuracy and better efficiency over existing 
CNNs, reducing parameter size and FLOPS by an order of magnitude. ... In particular, our EfficientNetB7 achieves new state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy, while being 8.4x smaller than the 
best existing CNN.
This project can be significantly used in multiple ways in the real world, especially in Hospitals and 
other Centre for tracking down and diagnosing with these diseases. One such use case can be the 
Hospital Management System where the Doctors and Medical Staff are trained and can diagnose the 
issues of the patient, in such a setting this model can help detect the disease of patients on call and give 
real-time diagnosis on the efficiency of the Doctors and Staff executive

## References

1. https://www.kaggle.com/artkulak/inference-45-55-600-epochs-tuned-effnet-b5-30-ep
2. https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/discussion/165727
3. https://www.kaggle.com/hfutybx/osic-feature-extract-from-ct
4. Benjamin Le Cook, Williard G. Manning, “Thinking beyond the mean: a practical guide for using 
quantile regression methods for health services research,” in US National library of Medicine, 2013.
5. Nancy L Wilczynski, Douglas Morgan, R Brian Haynes, the Hedges Team-1, “An overview of the 
design and methods for retrieving high-quality studies for clinical care,” in US National library of 
Medecine, 2005.
6. F.A. Gers, J. Schmidhuber, F. Cummins, “Learning to forget: continual prediction with LSTM,” in
IET: Digital Library , 1999.
7. https://www.kaggle.com/yasufuminakama/osic-lgb-baseline
8. Harvard Chen, “DICOM Processing and Segmentation in Python,” in Radiology Data Quest, 2017.
9. Alexander Etz, “Introduction to the Concept of Likelihood and Its Applications,” in Association of 
Psychological Science, 2018.
10. Nitish Srivastava and Geoffrey Hinton and Alex Krizhevsky and Ilya Sutskever and Ruslan 
Salakhutdinov, “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” in Journal 
of Machine Learning Research, 2014.
11. Sakshi Indolia, Anil Kumar Goswami, S.P. Mishra, Pooja Asopa,“ Conceptual 
understanding of Convolutional Neural Networks – A deep earning Approach” in
Procedia Computer Science, 2018.
xxxix
12. Simon L F Walsh, Stephen M Humphries, Athol U Wells, Kevin K Brown, “Imaging research in 
fibrotic lung disease; applying deep learning to unsolved problems”, in Lancet Respir Med, 2020.
13. He, K., Zhang, X., Ren, S., and Sun, J., “Deep residual learning for image recognition. 
CVPR,”
14. Walsh SLF, Calandriello L, Silva M, Sverzellati N., “Deep learning for classifying 
fibrotic lung disease on high-resolution computed tomography: a case-cohort study,” in 
Lancet Respir Med 2018.
15. Krizhevsky, A., Sutskever, I., and Hinton, G. E.,“ Imagenet classification with deep 
convolutional neural networks,” in NIPS, 2012.
16. M. Anthimopoulos, S. Christodoulidis, A. Christe and S. Mougiakakou, “ Classification 
of Interstitial Lung Disease Patterns Using Local DCT Features and Random Forest.”
17. Q. Li et al., "Lung image patch classification with automatic feature learning," in Proc. 
Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS 2013.
18. K.T. Vo et al., "Multiple kernel learning for classification of diffuse lung disease using 
HRCT lung images," in Proc. Int. Conf. IEEE Eng. Med. Biol. Soc. EMBS 2010
19. M. Gangeh et al., “A texton-based approach for the classification of lung parenchyma in 
ct images”, Med Image Comput Comput Assist Interv. Vol. 13(Pt 3)
20. Blackwell, Timothy S., et al. "Future directions in idiopathic pulmonary fibrosis research. 
An NHLBI workshop report." American journal of respiratory and critical care medicine 
189.2 (2014): 214-222.
21. Chua, Felix, Jack Gauldie, and Geoffrey J. Laurent. "Pulmonary fibrosis: searching for 
model answers." American journal of respiratory cell and molecular biology 33.1 (2005)
22. Czaplinski, A., A. A. Yen, and Stanley H. Appel. "Forced vital capacity (FVC) as an 
indicator of survival and disease progression in an ALS clinic population." Journal of 
Neurology, Neurosurgery & Psychiatry 77.3 (2006): 390-392.
23. Zappala, C. J., et al. "Marginal decline in forced vital capacity is associated with a poor 
outcome in idiopathic pulmonary fibrosis." European Respiratory Journal 35.4 (2010): 
xl
830-836.
24. Koenker, Roger, and Kevin F. Hallock. "Quantile regression." Journal of economic 
perspectives 15.4 (2001): 143-156.
25. Hao, Lingxin, Daniel Q. Naiman, and Daniel Q. Naiman. Quantile regression. No. 149. Sage, 2007.
26. Bottai, Matteo, Nicola Orsini, and Marco Geraci. "A gradient search maximization algorithm for the 
asymmetric Laplace likelihood." Journal of Statistical Computation and Simulation 85.10 (2015): 
1919-1925.
27. Baldi, Pierre, and Peter J. Sadowski. "Understanding dropout." Advances in neural information 
processing systems 26 (2013): 2814-2822
