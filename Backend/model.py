import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#% matplotlib inline

#data reading
wine_dataset = pd.read_csv('Dataset\winequality-red.csv')
#returning the matrix/table dimension -->(rows,cols)
wine_dataset.shape
#to show first 5 tuples
wine_dataset.head()
#returning description of data such as count 
wine_dataset.describe()
#plotting
#sns.catplot(x='quality', data=wine_dataset, kind='count')

#plot = plt.figure(figsize=(5,5))
#sns.barplot(x='quality', y= 'volatile acidity', data=wine_dataset)
#plt.pause(4)
#plt.show()

#corelation = wine_dataset.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(corelation, cbar=True, square=True, fmt= '.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

# dropping last column quality as it is label
X = wine_dataset.drop('quality', axis=1)

#taking qwality as label and classifying it in (0,1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

#splitting the dataset in train and test
# test size it 20 percent
# random state is shuffling 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#returning the no of tuples
print(Y.shape, Y_train.shape, Y_test.shape)

# MODEL TRAINING

#traing dataset using different decision trees in randomly selected subsets
model = RandomForestClassifier()
#we will make this decision trees from training set
model.fit(X_train,Y_train)

# MODEL EVALUATION

#ppredicting labels train sets using classifier and using it see accuracy 
X_train_prediction = model.predict(X_train)
train_data_accuracy  = accuracy_score(X_train_prediction, Y_train)
print(train_data_accuracy)

"""

# Building predictive model for hardcore input

#giving inputs
input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

# converting the data into numpy array
input_data_as_array = np.asarray(input_data)
#reshaping it as it have single feature
input_data_reshaped = input_data_as_array.reshape(1,-1)
#now doing prediction on input data
prediction = model.predict(input_data_reshaped)

#printing prediction


if (prediction[0] == 1):
    print('Good Wine Quality')
else:
    print('Bad wine Quality')

print(prediction)

"""
import pickle
#creating sav file
newfile = 'model.sav'
#loading out model
pickle.dump(model,open(newfile,'wb'))
load_model = pickle.load(open(newfile,'rb'))

 