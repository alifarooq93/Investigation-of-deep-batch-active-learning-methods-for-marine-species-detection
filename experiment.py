from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import os
import shutil
import random
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras import optimizers, layers
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD

from keras import backend as K
import cv2 as cv
from sklearn.model_selection import train_test_split
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import entropy

class Experiment:

    def __init__(self) -> None:
        self.modelname = 'Al_model_softmax'
        self.data_path = '/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_data/'
        self.test_data_path = '/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_test/'
        self.alldata_image_names = []
        self.alldata_images = []
        self.alldata_image_labels = []
        self.test_alldata_image_names = []
        self.test_alldata_images = []
        self.test_alldata_image_labels = []
        self.test_alldata_image_prob = []
        self.test_alldata_image_org_prob = []
        self.train_image_names = []
        self.train_images = []
        self.train_image_labels = []
        self.train_image_features = []
        self.train_image_pred_labels = []
        self.train_image_org_labels = []
        self.train_labels_sns = []
        self.test_image_names = []
        self.test_images = []
        self.test_image_features = []
        self.test_image_labels = []
        self.test_image_pred_labels = []
        self.test_entropy = []
        self.k = 3
        self.colors =['red','blue', 'green']
        self.label = ['jellyfish_aurelia', 'fish_cod', 'fish_unspecified']
        self.test_distances = []
        self.test_centers = []
        self.al_count = []
        self.kmeans_al_count = []
        self.selected_images_al = []
        self.train_cm_accuracy = []
        self.test_cm_accuracy = []
        self.kmeans_centroids = []
        self.kmeans_distances = []
        #self.test_data_preperation()
        #self.copy_random_files()
        #self.collect_data(self.data_path)
        #self.collect_test_data(self.test_data_path)
        #self.prepare_train_data()
        #ep = -1
        #self.retrain_inception_model(ep)
        #self.active_learning()
        self.al_count = np.load('/home/afarooq/ali/yolov5/exp_results/' + "al_count.npy")
        self.kmeans_al_count = np.load('/home/afarooq/ali/yolov5/exp_results/' + "kmeans_al_count.npy")
        self.selected_images_al = np.load('/home/afarooq/ali/yolov5/exp_results/' + "selected_images_al.npy")
        self.train_cm_accuracy = np.load('/home/afarooq/ali/yolov5/exp_results/' + "train_cm_accuracy.npy", allow_pickle=True)
        self.test_cm_accuracy = np.load('/home/afarooq/ali/yolov5/exp_results/' + "test_cm_accuracy.npy", allow_pickle=True)
        #self.plot_cm_accuracy()
        #self.active_learning_count()
        #self.image_accuracy_plot()
        self.image_class_plot()

    def image_class_plot(self):
        import os.path
        count0_images = []
        count1_images = []
        count2_images = []
        count0_images.append(100)
        count1_images.append(100)
        count2_images.append(100)
        self.selected_images_al = self.selected_images_al.tolist()
        #print(self.kmeans_al_count)
        for i in self.kmeans_al_count:
            images = self.selected_images_al[:i]
            del self.selected_images_al[:i]
            count0 = 0
            count1 = 0
            count2 = 0
            for j in images:
                if(os.path.exists('/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_data/0/' + j)):
                    count0 = count0 + 1
                elif(os.path.exists('/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_data/1/' + j)):
                    count1 = count1 + 1
                elif(os.path.exists('/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_data/2/' + j)):
                    count2 = count2 + 1
            count0_images.append(count0)
            count1_images.append(count1)
            count2_images.append(count2)
        print(count0_images)
        print(count1_images)
        print(count2_images)
        X = np.arange(14)
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(X + 0.00, count0_images, color = 'b', width = 0.25)
        ax.bar(X + 0.25, count1_images, color = 'g', width = 0.25)
        ax.bar(X + 0.50, count2_images, color = 'r', width = 0.25)
        ax.set_xlabel('No. of epcohs')
        ax.set_ylabel('No.of Images')
        ax.set_xticks(X)
        labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
        ax.set_xticklabels(labels)
        ax.set_title('No. of selected images per class')
        ax.legend(labels=['Jellyfish_aurelia', 'fish_cod', 'fish_unspecified'])
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/images_plot.png")

    def image_accuracy_plot(self):
        train_accuracy = []
        for i in self.train_cm_accuracy:
            train_accuracy.append(i[3])
        train_accuracy.append(0.9857978107707986)
        print(train_accuracy)
        print("acc count", len(train_accuracy))
        kmeans_count = []
        c = 300
        kmeans_count.append(c)
        for i in range(len(self.kmeans_al_count)):
            c = c + self.kmeans_al_count[i]
            kmeans_count.append(c)
        print(kmeans_count)
        print("km len", len(kmeans_count))
        plt.tight_layout()
        plt.xlabel('No.of Images')
        plt.ylabel('Accuracy')
        plt.plot(kmeans_count, train_accuracy, label = 'active learning accuracy w.r.t images')
        plt.legend()
        plt.title('active_learning_accuracy_with_images')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/active_learning_accuracy_w.r.t_images.png")
        plt.close()
        plt.cla()
        plt.clf()

    def plot_cm_accuracy(self):
        test_accuracy = []
        train_accuracy = []
        for i in self.test_cm_accuracy:
            test_accuracy.append(i[3])
        for i in self.train_cm_accuracy:
            train_accuracy.append(i[3])
        plt.tight_layout()
        plt.xlabel('epochs')
        plt.ylabel('accuracy_score')
        plt.plot(train_accuracy, label = 'train_accuracy')
        plt.plot(test_accuracy, label = 'test_accuracy')
        plt.legend()
        plt.title('train_test_accuracy')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/train_test_accuracy.png")
        plt.close()
        plt.cla()
        plt.clf()
        
    def active_learning_count(self):
        sns_count = []
        kmeans_count = []
        c = 300
        sns_count.append(c)
        for i in range(len(self.al_count)):
            c = c + self.al_count[i]
            sns_count.append(c)
        c = 300
        kmeans_count.append(c)
        for i in range(len(self.kmeans_al_count)):
            c = c + self.kmeans_al_count[i]
            kmeans_count.append(c)
        plt.tight_layout()
        plt.xlabel('epochs')
        plt.ylabel('no. of images')
        plt.plot(kmeans_count, label = 'active_learning_kmeans')
        plt.plot(sns_count, label = 'active_learning_sns')
        plt.legend()
        plt.title('active_learning_selection')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/active_learning_selection.png")
        plt.close()
        plt.cla()
        plt.clf()
        
    def test_data_preperation(self):
        src_dir = "/home/afarooq/ali/yolov5/data/datasets/ufo_al/train/1/"
        dest_dir = "/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_test/1/"
        shutil.copytree(src_dir, dest_dir)
        count = 0
        # Iterate directory
        for path in os.listdir(dest_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(dest_dir, path)):
                count += 1
        print('File count:', count)

    def copy_random_files(self):
        no_of_files = 1500
        source = "/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_test/2/"
        dest = "/home/afarooq/ali/yolov5/data/datasets/ufo_al/exp_data/2/"
        
        files = os.listdir(source)

        for file_name in random.sample(files, no_of_files):
            shutil.move(os.path.join(source, file_name), dest)
        
        count = 0
        # Iterate directory
        for path in os.listdir(dest):
            # check if current path is a file
            if os.path.isfile(os.path.join(dest, path)):
                count += 1
        print('File count:', count)

        count = 0
        # Iterate directory
        for path in os.listdir(source):
            # check if current path is a file
            if os.path.isfile(os.path.join(source, path)):
                count += 1
        print('File count:', count)

    def image_feature(self):
        self.loaded_model = keras.models.load_model(self.modelname)
        f = K.function(self.loaded_model.input, self.loaded_model.get_layer('mixed10').output)
        train_image = np.asarray(self.train_images)
        train_labels = np.asarray(self.train_image_labels)
        train_labels = tf.one_hot(train_labels, depth=3)
        feature_datagen =  ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        feature_datagen = feature_datagen.flow(train_image, train_labels, 1, shuffle = False)
        print("count", len(train_image))
        count = 0
        for f_dn in feature_datagen:
            count = count + 1
            x = f_dn[0]
            prob=self.loaded_model.predict(x)
            feat = f(x)
            feat=feat.flatten()
            self.train_image_features.append(feat)
            self.train_image_pred_labels.append(prob)
            if(count == len(train_image)):
                break
        print("features", len(self.train_image_features))
        print("pred_prob", len(self.train_image_pred_labels))

    def predicted_prob(self):
        for p in self.train_image_pred_labels:
            result = np.where(p[0] == max(p[0]))
            if (result[0][0] == 0):
                self.train_image_org_labels.append(0)
            elif (result[0][0] == 1):
                self.train_image_org_labels.append(1)
            elif (result[0][0] == 2):
                self.train_image_org_labels.append(2)
            else:
                print("same prob", result)

    def plot_confusion_matrix(self, y_test,y_scores, classNames, epoch):
        classes = len(classNames)
        cm = confusion_matrix(y_test, y_scores)
        print("**** Confusion Matrix ****")
        print(cm)
        print("**** Classification Report ****")
        print(classification_report(y_test, y_scores, target_names=classNames))
        res = classification_report(y_test, y_scores, target_names=classNames, output_dict = True)
        result = list(res.values())
        self.train_cm_accuracy.append(result)
        con = np.zeros((classes,classes))
        for x in range(classes):
            for y in range(classes):
                con[x,y] = cm[x,y]/np.sum(cm[x,:])

        plt.figure(figsize=(40,40))
        plt.tight_layout()
        sns.set(font_scale=3.0) # for label size
        df = sns.heatmap(con, annot=True,fmt='.2', cmap='Blues',xticklabels= classNames , yticklabels= classNames)
        df.set_xlabel("predicted label")
        df.set_ylabel("true label")
        df.set_title("confusion matrix")
        df.figure.savefig("/home/afarooq/ali/yolov5/exp_results/confusion_matrix " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def confusion_matrix(self, epoch):                   
        classNames = ['jellyfish_aurelia', 'fish_cod', 'fish_unspecified']
        self.plot_confusion_matrix(self.train_image_labels , self.train_image_org_labels , classNames, epoch)

    def kmeans(self, epoch):
        clusters = KMeans(self.k, random_state = 40)
        clusters.fit_predict(self.train_image_features)
        self.centroids = clusters.cluster_centers_
        self.labels = clusters.labels_
        print("centroids", len(self.centroids))
        print("labels", len(self.labels))
        image_distance = pd.DataFrame([])
        for center in range(len(self.centroids)):
            distance = []
            for feature in range(len(self.train_image_features)):
                temp = self.centroids[center] - self.train_image_features[feature]
                distance += [np.linalg.norm(temp)]
            image_distance["class"+str(center)] = distance
        self.class0_distance = image_distance.loc[:,'class0']
        self.class1_distance = image_distance.loc[:,'class1']
        self.class2_distance = image_distance.loc[:,'class2']
        self.class0_distance = self.class0_distance.values
        self.class1_distance = self.class1_distance.values
        self.class2_distance = self.class2_distance.values
        print("distance 0", len(self.class0_distance))
        print("distance 1", len(self.class1_distance))
        print("distance 2", len(self.class2_distance))
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_cod")
        plt.title("distance class jellyfish_aurelia and fish_cod")
        scatter = plt.scatter(self.class0_distance, self.class1_distance, c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/kmeans_class01 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class jellyfish_aurelia and fish_unspecified")
        scatter = plt.scatter(self.class0_distance, self.class2_distance, c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/kmeans_class02 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance fish_cod")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class fish_cod and fish_unspecified")
        scatter = plt.scatter(self.class1_distance, self.class2_distance, c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/kmeans_class12 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        
    def pca_kmeans(self, epoch):
        pca = PCA(n_components=100)
        principalComponents = pca.fit_transform(self.train_image_features)
        total = pca.explained_variance_ratio_.sum()
        print("total ratio", total)
        
        pca_clusters = KMeans(self.k, random_state = 40)
        pca_clusters.fit_predict(principalComponents)
        pca_centroids = pca_clusters.cluster_centers_

        pca_image_distance = pd.DataFrame([])
        for center in range(len(pca_centroids)):
            distance = []
            for feature in range(len(principalComponents)):
                temp = pca_centroids[center] - principalComponents[feature]
                distance += [np.linalg.norm(temp)]
            pca_image_distance["class"+str(center)] = distance
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_cod")
        plt.title("distance class jellyfish_aurelia and fish_cod")
        scatter = plt.scatter(pca_image_distance["class0"], pca_image_distance["class1"], c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/pca_kmeans_class01 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class jellyfish_aurelia and fish_unspecified")
        scatter = plt.scatter(pca_image_distance["class0"], pca_image_distance["class2"], c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/pca_kmeans_class02 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance fish_cod")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class fish_cod and fish_unspecified")
        scatter = plt.scatter(pca_image_distance["class1"], pca_image_distance["class2"], c=self.train_image_labels, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/pca_kmeans_class12 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def set_actual_labels_sns(self):
        for label in self.train_image_labels:
            if(label == 0):
                self.train_labels_sns.append('jellyfish_aurelia')
            elif(label == 1):
                self.train_labels_sns.append('fish_cod')
            elif(label == 2):
                self.train_labels_sns.append('fish_unspecified')

    def sns_plot(self, epoch):
        palette = sns.color_palette("bright", 3)
        tsne = TSNE()
        X_embedded = tsne.fit_transform(self.train_image_features)
        #print("x", len(X_embedded))
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=self.train_labels_sns, legend="full", palette=palette)
        plt.title("data_Clustering")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/sns_clusters_feature " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def pca_sns_plot(self, epoch):
        pca = PCA(n_components=100)
        principalComponents = pca.fit_transform(self.train_image_features)
        total = pca.explained_variance_ratio_.sum()
        print("total ratio", total)

        palette = sns.color_palette("bright", 3)
        tsne = TSNE()
        X_embedded = tsne.fit_transform(principalComponents)
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=self.train_labels_sns, legend="full", palette=palette)
        plt.title("data_Clustering")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/pca_sns_clusters_feature " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        
    def test_data(self):
        self.test_images.append(self.alldata_images[0][:100])
        self.test_image_names.append(self.alldata_image_names[0][:100])
        self.test_image_labels.append(self.alldata_image_labels[0][:100])
        self.test_images.append(self.alldata_images[1][:100])
        self.test_image_names.append(self.alldata_image_names[1][:100])
        self.test_image_labels.append(self.alldata_image_labels[1][:100])
        self.test_images.append(self.alldata_images[2][:100])
        self.test_image_names.append(self.alldata_image_names[2][:100])
        self.test_image_labels.append(self.alldata_image_labels[2][:100])
        del self.alldata_image_names[0][:100]
        del self.alldata_images[0][:100]
        del self.alldata_image_labels[0][:100]
        del self.alldata_image_names[1][:100]
        del self.alldata_images[1][:100]
        del self.alldata_image_labels[1][:100]
        del self.alldata_image_names[2][:100]
        del self.alldata_images[2][:100]
        del self.alldata_image_labels[2][:100]
        self.test_images = [item for sublist in self.test_images for item in sublist]
        self.test_image_names = [item for sublist in self.test_image_names for item in sublist]
        self.test_image_labels = [item for sublist in self.test_image_labels for item in sublist]
        print(len(self.test_image_labels))
        print(len(self.test_image_names))
        print(len(self.test_images))
        print(len(self.alldata_images[0]))
        print(len(self.alldata_images[1]))
        print(len(self.alldata_images[2]))
    
    def test_image_feature(self, epoch):
        self.loaded_model = keras.models.load_model(self.modelname)
        f = K.function(self.loaded_model.input, self.loaded_model.get_layer('mixed10').output)
        test_image = np.asarray(self.test_images)
        test_labels = np.asarray(self.test_image_labels)
        test_labels = tf.one_hot(test_labels, depth=3)
        feature_datagen =  ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        feature_datagen = feature_datagen.flow(test_image, test_labels, 1, shuffle = False)
        print("count", len(test_image))
        count = 0
        for f_dn in feature_datagen:
            count = count + 1
            x = f_dn[0]
            prob=self.loaded_model.predict(x)
            feat = f(x)
            feat=feat.flatten()
            self.test_image_features.append(feat)
            self.test_image_pred_labels.append(prob)
            if(count == len(test_image)):
                break
        print("features", len(self.test_image_features))
        print("pred_prob", len(self.test_image_pred_labels))

        test_f = self.train_image_features
        test_f = test_f + self.test_image_features

        test_l = self.train_image_labels
        test_l = test_l + self.test_image_labels

        test_p = self.train_image_pred_labels
        test_p = test_p + self.test_image_pred_labels

        print("test features", len(test_f))
        print("test image_label", len(test_l))
        print("test image_prob", len(test_p))
        
        clusters = KMeans(self.k, random_state = 40)
        clusters.fit_predict(test_f)
        self.kmeans_centroids = clusters.cluster_centers_
        print("centroids", len(self.centroids))
        self.kmeans_distances = test_f[-300:]
        image_distance = pd.DataFrame([])
        for center in range(len(self.kmeans_centroids)):
            distance = []
            for feature in range(len(test_f)):
                temp = self.kmeans_centroids[center] - test_f[feature]
                distance += [np.linalg.norm(temp)]
            image_distance["class"+str(center)] = distance
        self.class0_distance = image_distance.loc[:,'class0']
        self.class1_distance = image_distance.loc[:,'class1']
        self.class2_distance = image_distance.loc[:,'class2']
        self.class0_distance = self.class0_distance.values
        self.class1_distance = self.class1_distance.values
        self.class2_distance = self.class2_distance.values
        print("distance 0", len(self.class0_distance))
        print("distance 1", len(self.class1_distance))
        print("distance 2", len(self.class2_distance))
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_cod")
        plt.title("distance class jellyfish_aurelia and fish_cod")
        scatter = plt.scatter(self.class0_distance, self.class1_distance, c=test_l, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/test_kmeans_class01 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance jellyfish_aurelia")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class jellyfish_aurelia and fish_unspecified")
        scatter = plt.scatter(self.class0_distance, self.class2_distance, c=test_l, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/test_kmeans_class02 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("Distance fish_cod")
        plt.ylabel("Distance fish_unspecified")
        plt.title("distance class fish_cod and fish_unspecified")
        scatter = plt.scatter(self.class1_distance, self.class2_distance, c=test_l, cmap=matplotlib.colors.ListedColormap(self.colors))
        plt.legend(handles=scatter.legend_elements()[0], labels=self.label)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/test_kmeans_class12 " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

        self.test_labels_sns = []
        for label in test_l:
            if(label == 0):
                self.test_labels_sns.append('jellyfish_aurelia')
            elif(label == 1):
                self.test_labels_sns.append('fish_cod')
            elif(label == 2):
                self.test_labels_sns.append('fish_unspecified')

        print("unique", set(self.test_labels_sns))
        palette = sns.color_palette("bright", 3)
        tsne = TSNE()
        X_embedded = tsne.fit_transform(test_f)
        self.test_distances = X_embedded[-300:]
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=self.test_labels_sns, legend="full", palette=palette)
        plt.title("test_data_Clustering")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/clusters_with_new_images " + str(epoch) + ".png")
        kmeans = KMeans(self.k).fit(X_embedded)
        self.test_centers = kmeans.cluster_centers_
        sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], palette=palette, s=40, ec='black')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/clusters_with_new_images_centers " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def entropy_plot(self, epoch):
        plt.figure(figsize=(30,20))
        plt.tight_layout()
        plt.xlabel("entropy score")
        plt.ylabel("test images")
        plt.title("entropy scores for test images")
        plt.hist(self.test_entropy)
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/entropy_score " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def test_images_entropy(self):
        for p in self.test_image_pred_labels:
            ent = entropy(p[0], base=2)
            self.test_entropy.append(ent)

    def al_images(self):
        image_distance = pd.DataFrame([])
        for center in range(len(self.test_centers)):
            distance = []
            for feature in range(len(self.test_distances)):
                temp = self.test_centers[center] - self.test_distances[feature]
                distance += [np.linalg.norm(temp)]
            image_distance["class"+str(center)] = distance
        self.class0_distance = image_distance.loc[:,'class0']
        self.class1_distance = image_distance.loc[:,'class1']
        self.class2_distance = image_distance.loc[:,'class2']
        self.class0_distance = self.class0_distance.values
        self.class1_distance = self.class1_distance.values
        self.class2_distance = self.class2_distance.values
        print("distance 0", len(self.class0_distance))
        print("distance 1", len(self.class1_distance))
        print("distance 2", len(self.class2_distance))
        al_count = 0
        for i in range(300):
            label = self.test_image_labels[i]
            if label == 0:
                if(self.class0_distance[i] >= self.class1_distance[i] or self.class0_distance[i] >= self.class2_distance[i] or self.test_entropy[i] > 0.90):
                    al_count = al_count + 1
            elif label == 1:
                if(self.class2_distance[i] >= self.class0_distance[i] or self.class2_distance[i] >= self.class1_distance[i] or self.test_entropy[i] > 0.90):
                    al_count = al_count + 1
            elif label == 2:
                if(self.class1_distance[i] >= self.class0_distance[i] or self.class1_distance[i] >= self.class2_distance[i] or self.test_entropy[i] > 0.90):
                    al_count = al_count + 1
        print("sns_selected_images", al_count)
        self.al_count.append(al_count)

    def al_kmeans_images(self):
        image_distance = pd.DataFrame([])
        for center in range(len(self.kmeans_centroids)):
            distance = []
            for feature in range(len(self.kmeans_distances)):
                temp = self.kmeans_centroids[center] - self.kmeans_distances[feature]
                distance += [np.linalg.norm(temp)]
            image_distance["class"+str(center)] = distance
        self.class0_distance = image_distance.loc[:,'class0']
        self.class1_distance = image_distance.loc[:,'class1']
        self.class2_distance = image_distance.loc[:,'class2']
        self.class0_distance = self.class0_distance.values
        self.class1_distance = self.class1_distance.values
        self.class2_distance = self.class2_distance.values
        print("distance 0", len(self.class0_distance))
        print("distance 1", len(self.class1_distance))
        print("distance 2", len(self.class2_distance))
        al_count = 0
        for i in range(300):
            label = self.test_image_labels[i]
            if label == 0:
                if(self.class0_distance[i] >= self.class1_distance[i] or self.class0_distance[i] >= self.class2_distance[i] or self.test_entropy[i] > 0.90):
                    self.train_image_names.append(self.test_image_names[i])
                    self.train_images.append(self.test_images[i])
                    self.train_image_labels.append(self.test_image_labels[i])
                    self.selected_images_al.append(self.test_image_names[i])
                    al_count = al_count + 1
            elif label == 1:
                if(self.class2_distance[i] >= self.class0_distance[i] or self.class2_distance[i] >= self.class1_distance[i] or self.test_entropy[i] > 0.90):
                    self.train_image_names.append(self.test_image_names[i])
                    self.train_images.append(self.test_images[i])
                    self.train_image_labels.append(self.test_image_labels[i])
                    self.selected_images_al.append(self.test_image_names[i])
                    al_count = al_count + 1
            elif label == 2:
                if(self.class1_distance[i] >= self.class0_distance[i] or self.class1_distance[i] >= self.class2_distance[i] or self.test_entropy[i] > 0.90):
                    self.train_image_names.append(self.test_image_names[i])
                    self.train_images.append(self.test_images[i])
                    self.train_image_labels.append(self.test_image_labels[i])
                    self.selected_images_al.append(self.test_image_names[i])
                    al_count = al_count + 1
        print("kmeans_selected_images", al_count)
        self.kmeans_al_count.append(al_count)

    def testing_image_features(self):
        self.loaded_model = keras.models.load_model(self.modelname)
        test_image = np.asarray(self.test_alldata_images)
        test_labels = np.asarray(self.test_alldata_image_labels)
        test_labels = tf.one_hot(test_labels, depth=3)
        feature_datagen =  ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        feature_datagen = feature_datagen.flow(test_image, test_labels, 1, shuffle = False)
        print("count", len(test_image))
        count = 0
        for f_dn in feature_datagen:
            count = count + 1
            x = f_dn[0]
            prob=self.loaded_model.predict(x)
            self.test_alldata_image_prob.append(prob)
            if(count == len(test_image)):
                break
        print("pred_prob", len(self.test_alldata_image_prob))

    def testing_predicted_prob(self):
        for p in self.test_alldata_image_prob:
            result = np.where(p[0] == max(p[0]))
            if (result[0][0] == 0):
                self.test_alldata_image_org_prob.append(0)
            elif (result[0][0] == 1):
                self.test_alldata_image_org_prob.append(1)
            elif (result[0][0] == 2):
                self.test_alldata_image_org_prob.append(2)
            else:
                print("same prob", result)

    def plot_testing_confusion_matrix(self, y_test, y_scores, classNames, epoch):
        classes = len(classNames)
        cm = confusion_matrix(y_test, y_scores)
        print("**** Confusion Matrix ****")
        print(cm)
        print("**** Classification Report ****")
        print(classification_report(y_test, y_scores, target_names=classNames))
        res = classification_report(y_test, y_scores, target_names=classNames, output_dict = True)
        result = list(res.values())
        self.test_cm_accuracy.append(result)
        con = np.zeros((classes,classes))
        for x in range(classes):
            for y in range(classes):
                con[x,y] = cm[x,y]/np.sum(cm[x,:])

        plt.figure(figsize=(40,40))
        plt.tight_layout()
        sns.set(font_scale=3.0) # for label size
        df = sns.heatmap(con, annot=True,fmt='.2', cmap='Blues',xticklabels= classNames , yticklabels= classNames)
        df.set_xlabel("predicted label")
        df.set_ylabel("true label")
        df.set_title("confusion matrix")
        df.figure.savefig("/home/afarooq/ali/yolov5/exp_results/testing_confusion_matrix " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def testing_confusion_matrix(self, epoch):
        classNames = ['jellyfish_aurelia', 'fish_cod', 'fish_unspecified']
        self.plot_testing_confusion_matrix(self.test_alldata_image_labels , self.test_alldata_image_org_prob , classNames, epoch)

    def active_learning(self):
        for i in range(13):
            self.image_feature()
            self.predicted_prob()
            self.confusion_matrix(i)
            self.kmeans(i)
            self.pca_kmeans(i)
            self.set_actual_labels_sns()
            self.sns_plot(i)
            self.pca_sns_plot(i)
            self.test_data()
            self.test_image_feature(i)
            self.test_images_entropy()
            self.entropy_plot(i)
            self.al_images()
            self.al_kmeans_images()
            print(len(self.train_image_labels))
            print(len(self.train_image_names))
            print(len(self.train_images))
            self.retrain_inception_model(i)
            self.testing_image_features()
            self.testing_predicted_prob()
            self.testing_confusion_matrix(i)
            self.train_image_features = []
            self.train_image_pred_labels = []
            self.train_image_org_labels = []
            self.train_labels_sns = []
            self.test_image_names = []
            self.test_images = []
            self.test_image_features = []
            self.test_image_labels = []
            self.test_image_pred_labels = []
            self.test_entropy = []
            self.test_distances = []
            self.test_centers = []
            self.kmeans_centroids = []
            self.kmeans_distances = []
            self.test_alldata_image_prob = []
            self.test_alldata_image_org_prob = []
        np.save("/home/afarooq/ali/yolov5/exp_results/al_count.npy", self.al_count)
        np.save("/home/afarooq/ali/yolov5/exp_results/kmeans_al_count.npy", self.kmeans_al_count)
        np.save("/home/afarooq/ali/yolov5/exp_results/selected_images_al.npy", self.selected_images_al)
        np.save("/home/afarooq/ali/yolov5/exp_results/train_cm_accuracy.npy", self.train_cm_accuracy)
        np.save("/home/afarooq/ali/yolov5/exp_results/test_cm_accuracy.npy", self.test_cm_accuracy)

    def retrain_inception_model(self, epoch):
        pre_model = InceptionV3(weights='imagenet', include_top=False, input_shape = (150, 150, 3))

        x = pre_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation="relu", name = "features1")(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation="relu", name = "features2")(x)
        predictions = Dense(3, activation="softmax")(x)

        self.model_ = Model(inputs=pre_model.input, outputs=predictions)

        for layer in self.model_.layers[:52]:
            layer.trainable = False

        # compile the model
        self.model_.compile(optimizer=SGD(lr=0.0001, momentum=0.9)
                            , loss='categorical_crossentropy'
                            , metrics=['accuracy'])

        train_datagen =  ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        val_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.train_images
                                                                            , self.train_image_labels
                                                                            , test_size = 0.15
                                                                            , random_state = 41)
        
        self.x_train = np.asarray(self.x_train)
        self.y_train = np.asarray(self.y_train)
        self.y_train = tf.one_hot(self.y_train, depth=3)

        self.x_val = np.asarray(self.x_val)
        self.y_val = np.asarray(self.y_val)
        self.y_val = tf.one_hot(self.y_val, depth=3)

        train_generator = train_datagen.flow(self.x_train, self.y_train, 20)
        val_generator = val_datagen.flow(self.x_val, self.y_val, 20)

        self.hist = self.model_.fit(train_generator
                            , validation_data = val_generator
                            , epochs= 50
                            , verbose = 1)
 
        self.model_.save(self.modelname)

        self.check_accuracy(epoch)
        self.check_loss(epoch)

    def check_accuracy(self, epoch):
        plt.figure(figsize=(20, 20))
        plt.tight_layout()
        plt.plot(self.hist.history['accuracy'], label = 'train_accuracy')
        plt.plot(self.hist.history['val_accuracy'], label = 'val_accuracy')
        plt.legend()
        plt.title('Accuracy')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/train_val_accuracy " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def check_loss(self, epoch):
        plt.figure(figsize=(20, 20))
        plt.tight_layout()
        plt.plot(self.hist.history['loss'], label = 'train_loss')
        plt.plot(self.hist.history['val_loss'], label = 'valid_loss')
        plt.legend()
        plt.title('Loss')
        plt.savefig("/home/afarooq/ali/yolov5/exp_results/train_val_loss " + str(epoch) + ".png")
        plt.close()
        plt.cla()
        plt.clf()

    def prepare_train_data(self):
        self.train_images.append(self.alldata_images[0][:200])
        self.train_image_names.append(self.alldata_image_names[0][:200])
        self.train_image_labels.append(self.alldata_image_labels[0][:200])
        self.train_images.append(self.alldata_images[1][:200])
        self.train_image_names.append(self.alldata_image_names[1][:200])
        self.train_image_labels.append(self.alldata_image_labels[1][:200])
        self.train_images.append(self.alldata_images[2][:200])
        self.train_image_names.append(self.alldata_image_names[2][:200])
        self.train_image_labels.append(self.alldata_image_labels[2][:200])
        del self.alldata_image_names[0][:200]
        del self.alldata_images[0][:200]
        del self.alldata_image_labels[0][:200]
        del self.alldata_image_names[1][:200]
        del self.alldata_images[1][:200]
        del self.alldata_image_labels[1][:200]
        del self.alldata_image_names[2][:200]
        del self.alldata_images[2][:200]
        del self.alldata_image_labels[2][:200]
        self.train_images = [item for sublist in self.train_images for item in sublist]
        self.train_image_names = [item for sublist in self.train_image_names for item in sublist]
        self.train_image_labels = [item for sublist in self.train_image_labels for item in sublist]

    def collect_data(self, dataset_path):
        for directory in os.listdir(dataset_path):
            alldata_images = []
            alldata_image_names = []
            alldata_image_labels = []
            for file in os.listdir(os.path.join(dataset_path, directory)):
                image_path = os.path.join(dataset_path, directory, file)
                image = cv.imread(image_path, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (150, 150), interpolation = cv.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                alldata_images.append(image)
                alldata_image_names.append(file)
                alldata_image_labels.append(int(directory))
            self.alldata_images.append(alldata_images)
            self.alldata_image_names.append(alldata_image_names)
            self.alldata_image_labels.append(alldata_image_labels)

    def collect_test_data(self, dataset_path):
        for directory in os.listdir(dataset_path):
            alldata_images = []
            alldata_image_names = []
            alldata_image_labels = []
            for file in os.listdir(os.path.join(dataset_path, directory)):
                image_path = os.path.join(dataset_path, directory, file)
                image = cv.imread(image_path, cv.COLOR_BGR2RGB)
                image = cv.resize(image, (150, 150), interpolation = cv.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                alldata_images.append(image)
                alldata_image_names.append(file)
                alldata_image_labels.append(int(directory))
            self.test_alldata_images.append(alldata_images)
            self.test_alldata_image_names.append(alldata_image_names)
            self.test_alldata_image_labels.append(alldata_image_labels)
        self.test_alldata_images = [item for sublist in self.test_alldata_images for item in sublist]
        self.test_alldata_image_names = [item for sublist in self.test_alldata_image_names for item in sublist]
        self.test_alldata_image_labels = [item for sublist in self.test_alldata_image_labels for item in sublist]
        print(len(self.test_alldata_images))
        print(len(self.test_alldata_image_names))
        print(len(self.test_alldata_image_labels))

def main():
    cs = Experiment()
    
if __name__ == "__main__":
    main()