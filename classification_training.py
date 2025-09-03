#This notebook is only executed once to train the classification algorithms outside of the app
#for better performance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap

data = load_breast_cancer(as_frame = True)

cm = plt.cm.RdBu_r
cm_bright = ListedColormap(["#0000FF","#FF0000"])


def train_clf_bcw(features = None): 
    if features: 

        data_points = data['data'][[features[0], features[1]]].to_numpy()
        layers = (30,5) #we use a simple 2 layer MLP because this is just a binary classification now with 2 features
    else: 
        data_points = data['data'].to_numpy()
        layers = (100,15,)
    labels = data['target'].to_numpy()


    x_train, x_test, y_train, y_test = train_test_split(data_points, labels, test_size = 0.2, random_state= 42)

    #we have 30 features , so we increase the size and than decrease the size again
    clf = MLPClassifier(hidden_layer_sizes = layers, random_state=42, max_iter = 500 ).fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = clf.score(x_test, y_test)

    alpha_error = np.sum(y_pred[np.nonzero(y_test == 0)] == 1)/np.sum(y_pred == 1)
    beta_error = np.sum(y_pred[np.nonzero(y_test == 1)] == 0)/np.sum(y_pred == 0)

    scores = np.zeros([2,2])
    scores[0,:] = [1-alpha_error, beta_error]
    scores[1,:] = [alpha_error, 1-beta_error]



    return [acc,scores], clf


def plot_clf(clf): 
    #get a linescore for 2 dims
    if features: 
        step = 0.02
        x_decs, y_decs = np.meshgrid(np.arange(data_points[:,0].min()-0.5, data_points[:,0].max()+0.5,step), np.arange(data_points[:,1].min()-0.5, data_points[:,1].max()+0.5,step))
        decs = clf.predict_proba(np.column_stack([x_decs.ravel(), y_decs.ravel()]))[:,1]
        decs = decs.reshape(x_decs.shape)

        fig, ax = plt.subplots(1,1)
        ax.contourf(x_decs, y_decs, decs, cmap = 'RdBu_r', alpha = 0.75 )
        scatter = ax.scatter(data_points[:,0],data_points[:,1], c = labels, cmap = cm_bright , edgecolors = 'k')
        handles = scatter.legend_elements()[0]
        print(handles)
        ax.legend(handles, ['malignant','benign',],bbox_to_anchor=(1.0, 1.0), loc='upper left',  frameon = False)
        plt.show()


    #fig1, ax1 = plt.subplots(1,1)
    #ax1.plot(clf.loss_curve_)
    #plt.show()

    print(acc)
    
    print('Dont have cancer, diganosed with cancer: '+str(alpha_error))
    print('Has cancer, not diagnosed with cancer: '+str(beta_error))

train_clf_bcw(features = ['mean radius', 'mean texture'])