#This notebook is only executed once to train the classification algorithms outside of the app
#for better performance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from simple_AI_examples import gen_data_clustering

from matplotlib.colors import ListedColormap
#for plotting 3 classes
import matplotlib.style as style
