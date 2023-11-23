# CS50 FINAL PROJECT 

This is going to be a machine learning repository, appliying some basics models to datasets that are interestings.

## Digit recognition

The first dataset analysis included is, digit handwritting recognition. 
In this project, I'm gonna analyze a dataset of number recognition, in order to build a model capable of identifying a number with an image as an output. <br>
The first thing we are going to do is import all the necessary things for this project including the dataset and all the Python libraries to use.
<br>
The dataset is provided by a library in python called **sckit learn**, which is a Python library dedicated to machine learning. We are gonna use the digits dataset. 

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
```
### Evaluating performance 

<ul>
    <li> Name: k-means++ <br> 
    <li>Time: 0.7189590930938721 <br>
    <li>inertia: 69519.937 <br>
    <li>Homogeneity score: 0.611 <br>
    <li>Completness score: 0.659 <br>
    <li>V measure score: 0.634 <br>
    <li>Adjusted rand score: 0.480 <br>
    <li>Adjusted rand score: 0.630 <br>
    <li>Silhouette score: 0.135 <br>
</ul>

## Age Estimation

## Predicting diabetes
