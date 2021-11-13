import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly as plt
import plotly.express as px
from django_plotly_dash import DjangoDash
import numpy as np
import os
from pydicom import dcmread
from pydicom import multival
from zipfile import ZipFile
import requests
import json
import tensorflow as tf
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import pandas as pd
from sklearn.cluster import KMeans
from skimage import morphology
import scipy.ndimage
from skimage import measure
from skimage.transform import resize


class Dash():
    def __init__(self, p):
        self.id = p.id
        self.name = p.name.split()[0]
        self.zip_path = p.dicoms.path
        self.unzip(self.zip_path)
        self.dcm_path = f'home/dicoms/{self.id}'

        self.features = {}

        self.features['baseweek'] = int(p.base_week)
        self.features['basepercent'] = int(p.base_percent)
        self.features['basefvc'] = int(p.base_fvc)
        self.features['currentweek'] = self.features['baseweek'] + 3
        self.features['diff'] = self.features['currentweek'] - \
            self.features['baseweek']
        self.features['age'] = int(p.age)
        self.features['sex'] = int(p.sex)
        self.features['smoking'] = int(p.smoking)

        self.get_features()

        for key in self.features.keys():
            self.features[key] = [self.features[key]]

        self.features = pd.DataFrame(self.features)
        self.features = pd.concat([self.features]*50, ignore_index = True)

        for i in range(1, 50):
            self.features['currentweek'][i] = self.features['currentweek'][i-1]+1
            self.features['diff'][i] = self.features['diff'][i-1]+1

        for col in self.features.columns:
            self.features[col] = np.asarray(
                self.features[col], dtype = np.float32)

        self.make_predictions()

        self.make_layout()

    def get_features(self):
        scan = self.get_scan(self.dcm_path, 0)

        self.features['area'] = float(scan.Columns*scan.Rows)

        spacing = scan.PixelSpacing
        self.features['psr'] = float(spacing[0])
        self.features['psc'] = float(spacing[1])

        self.features['st'] = float(scan.SliceThickness)

        self.features['ww'] = float(self.get_window_value(scan.WindowWidth))
        self.features['wl'] = float(self.get_window_value(scan.WindowCenter))
        self.features['rd'] = float(self.features['psr'] * scan.Rows)
        self.features['cd'] = float(self.features['psc'] * scan.Columns)
        self.features['acm'] = float(
            0.1 * self.features['rd'] * 0.1 * self.features['cd'])
        self.features['volcm'] = float(
            0.1 * self.features['st'] * self.features['acm'])

        self.get_eff_features()

    def get_eff_features(self):
        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']

        REQUEST_TIMEOUT = 200

        images = self.get_images(self.get_scans(self.dcm_path))
        images = self.normalize(self.crop_images(images))

        for i in range(len(images)):
            try:
                images[i] = images[i] * self.make_lungmask(images[i], False)
            except:
                images[i] = images[i] * np.zeros((512, 512))

        images = self.dimension_scaling(images)

        channel = grpc.insecure_channel("0.0.0.0:9000")
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'effnet'
        request.model_spec.signature_name = 'serving_default'
        tensor = tf.make_tensor_proto(images, dtype = tf.float32)

        request.inputs['input_1'].CopyFrom(tensor)

        result = stub.Predict(request, REQUEST_TIMEOUT)
        result = list(result.outputs['dense_1'].float_val)

        eff_features = []

        for i in range(0, len(result), 7):
            eff_features.append(result[i:i+7])

        eff_features = np.mean(eff_features, axis = 0)

        count = 0
        for a in eff_features:
            self.features[f"e{count}"] = a
            count += 1

    def make_predictions(self):
        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']

        REQUEST_TIMEOUT = 200

        channel = grpc.insecure_channel("0.0.0.0:9001")
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'lstmnet'
        request.model_spec.signature_name = 'serving_default'

        data = np.asarray(self.features.values)
        data = data.reshape((len(data), 25, 1))

        tensor = tf.make_tensor_proto(data, dtype = tf.float32)

        request.inputs['input_28'].CopyFrom(tensor)

        result = stub.Predict(request, REQUEST_TIMEOUT)
        result = list(result.outputs['preds'].float_val)

        self.preds = []

        for i in range(0, len(result), 3):
            self.preds.append(result[i:i+3])

        self.preds = np.asarray(self.preds)

    def unzip(self, path):
        os.makedirs(f'home/dicoms/{self.id}')

        with ZipFile(path, 'r') as zip:
            zip.extractall(f'home/dicoms/{self.id}')

    def get_window_value(self, feature):
        if type(feature) == multival.MultiValue:
            return np.int(feature[0])
        else:
            return np.int(feature)

    def get_scan(self, path, index):
        dicom = dcmread(os.path.join(path, os.listdir(path)[0]))

        return dicom

    def get_scans(self, path):
        dicoms = []

        for filename in os.listdir(path):
            dicoms.append(dcmread(os.path.join(path, filename)))

        return dicoms

    def get_images(self, scans):
        temp = []

        for i in range(len(scans)):
            try:
                temp.append(scans[i].pixel_array)
            except:
                return None

        return np.asarray(temp, dtype = np.int16)

    def crop_image(self, image):
        if image.shape != (512, 512):
            left = ((image.shape[0]-512)//2)
            right = image.shape[0]-left
            top = ((image.shape[1]-512)//2)
            bottom = image.shape[1]-top

            return image[left-1:right-1, top-1:bottom-1]
        else:
            return image

    def crop_images(self, images):
        if images[0].shape != (512, 512):
            temp = []
            for image in images:
                temp.append(self.crop_image(image))

            return np.asarray(temp)
        else:
            return images

    def dimension_scaling(self, images):
        '''
        Returns images with channels. e.g. converts 512x512 images to 512x512x3

        Inputs:
        imageslices:  NumpyArray  Imageslices extracted from the DICOM files using fnc get_images_from_slices

        '''
        images = np.expand_dims(images, axis = -1)
        images = np.concatenate((images, )*3, axis = -1)

        return np.asarray(images)

    def normalize(self, arr):
        return (arr-arr.min())/(arr.max()-arr.min())

    def make_lungmaskself(self, img, display = False):
        row_size = img.shape[0]
        col_size = img.shape[1]

        mean = np.mean(img)
        std = np.std(img)
        img = img-mean
        img = img/std
        # Find the average pixel value near the lungs
        # to renormalize washed out images
        middle = img[int(col_size/5):int(col_size/5*4), 
                     int(row_size/5):int(row_size/5*4)]
        mean = np.mean(middle)
        max = np.max(img)
        min = np.min(img)
        # To improve threshold finding, I'm moving the
        # underflow and overflow on the pixel spectrum
        img[img == max] = mean
        img[img == min] = mean
        #
        # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
        #
        kmeans = KMeans(n_clusters = 2).fit(
            np.reshape(middle, [np.prod(middle.shape), 1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

        # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
        # We don't want to accidentally clip the lung.

        eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
        dilation = morphology.dilation(eroded, np.ones([8, 8]))

        # Different labels are displayed in different colors
        labels = measure.label(dilation)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0] < row_size/10*9 and B[3]-B[1] < col_size/10*9 and B[0] > row_size/5 and B[2] < col_size/5*4:
                good_labels.append(prop.label)
        mask = np.ndarray([row_size, col_size], dtype = np.int8)
        mask[:] = 0

        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask
        #
        for N in good_labels:
            mask = mask + np.where(labels == N, 1, 0)
        mask = morphology.dilation(
            mask, np.ones([10, 10]))  # one last dilation

        if (display):
            fig, ax = plt.subplots(3, 2, figsize = [12, 12])
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(img, cmap = 'gray')
            ax[0, 0].axis('off')
            ax[0, 1].set_title("Threshold")
            ax[0, 1].imshow(thresh_img, cmap = 'gray')
            ax[0, 1].axis('off')
            ax[1, 0].set_title("After Erosion and Dilation")
            ax[1, 0].imshow(dilation, cmap = 'gray')
            ax[1, 0].axis('off')
            ax[1, 1].set_title("Color Labels")
            ax[1, 1].imshow(labels)
            ax[1, 1].axis('off')
            ax[2, 0].set_title("Final Mask")
            ax[2, 0].imshow(mask, cmap = 'gray')
            ax[2, 0].axis('off')
            ax[2, 1].set_title("Apply Mask on Original")
            ax[2, 1].imshow(mask*img, cmap = 'gray')
            ax[2, 1].axis('off')

            plt.show()

        return mask

    def make_graph(self):
        temp = np.mean(self.preds[:, 1])

        fig = px.line(y = temp, x = range(3, 53), 
                      title = 'FVC Prediction')
        fig.update_layout(
            autosize = True, 
            xaxis = {'title': 'Weeks'}, 
            yaxis = {'title': 'FVC score'}
        )

        return fig

    def make_layout(self):
        app = DjangoDash('dashapp', external_stylesheets = [dbc.themes.BOOTSTRAP])
        

        app.layout = dbc.Container([

            dbc.Row
            ([
                
                dbc.Col( 
                    dcc.Graph(figure = self.make_graph()), md = 12), 
            ], 
            align = "left", 
        
            ), 
                
        ], 
        )
