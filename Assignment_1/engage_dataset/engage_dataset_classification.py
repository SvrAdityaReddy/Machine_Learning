
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from pandas.plotting import scatter_matrix


# In[2]:


data=pd.read_csv("D49.csv")
data.head()


# In[3]:


data=data.drop('Unnamed: 0',axis=1)
data.head()


# # Exploratory Data Analysis

# In[4]:


data1=data.drop('Label',axis = 1)
data1.describe()


# In[5]:


data1.hist(bins=25,figsize=(20,15))
plt.show()


# In[6]:


data1=np.array(data1)
plt.figure(1,figsize=(15, 10))
plt.boxplot(data1)


# # Creating Training and Test Dataset

# In[7]:


train_set, test_set=train_test_split(data,test_size=0.2,random_state=42)
train_data=data.copy()


# In[8]:


correlation_matrix=train_data.corr()
correlation_matrix['Label'].sort_values(ascending=False)


# In[9]:


attributes = [ "Label", "F0","F1","F2","F3","F4","F5"]
scatter_matrix(train_data[attributes], figsize=(12, 8))


# # Preparing the Data

# In[10]:


train_data = train_set.drop('Label', axis=1)
train_labels = train_set['Label'].copy()
train_data.head()


# In[11]:


test_data = test_set.drop('Label', axis=1)
test_labels = test_set['Label'].copy()
test_data.shape


# ## Checking for any null or NAN values in training dataset

# In[12]:


sample_incomplete_rows = train_data[train_data.isnull().any(axis=1)].head()
sample_incomplete_rows


# ## Checking for any null or NAN values in test dataset

# In[13]:


sample_incomplete_rows = test_data[test_data.isnull().any(axis=1)].head()
sample_incomplete_rows


# # Binary Classification

# In[14]:


X=np.array(train_data)
X=X[:,(3,4,5)]
Y=np.array(train_labels).flatten()
test_data=np.array(test_data)
test_data=test_data[:,(3,4,5)]
test_labels=np.array(test_labels).flatten()


# # Nearest Neighbours

# In[15]:


predicted_labels=[]
for i in range(len(test_data)):
    # euclidean distance
    minimum_distance=((np.dot(test_data[i],test_data[i]))-2*(np.dot(test_data[i],X[0]))+(np.dot(X[0],X[0])))**0.5
    closest_neighbour=Y[0]
    for j in range(1,len(X)):
        # euclidean distance
        distance=((np.dot(test_data[i],test_data[i]))-2*(np.dot(test_data[i],X[j]))+(np.dot(X[j],X[j])))**0.5
        if(distance < minimum_distance):
            minimum_distance=distance
            closest_neighbour=Y[j]
    predicted_labels.append(closest_neighbour)
# print(predicted_labels)


# # Accuracy score - Nearest Neighbours

# In[16]:


metrics.accuracy_score(test_labels,predicted_labels)


# # Precision - Nearest Neighbours

# In[17]:


metrics.precision_score(test_labels,predicted_labels)


# # F-measure - Nearest Neighbours

# In[18]:


metrics.f1_score(test_labels, predicted_labels)


# # Recall - Nearest Neighbours

# In[19]:


metrics.recall_score(test_labels,predicted_labels)


# # AUC - Nearest Neighbours

# In[20]:


fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
metrics.auc(fpr, tpr)


# # Naive Bayes Classifier

# In[21]:


# Assuming data is fitted to a Gaussian
def probability(mean, std, x):
    exponential=np.exp(-1*(x-mean)**2/(2*(std**2)))
    return ((1/(std*((22/7.0)**0.5)))*(exponential))


# In[22]:


# Fitting Gausian
def gaussian_parameters(X):
    mean=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    return (mean,std)


# The following code is to get data points corresponding to each class

# In[23]:


data_class1= [X[i] for i in range(len(Y)) if Y[i]==1] # class1 refers to data corresponding to flower Iris-Virginica
data_class2= [X[i] for i in range(len(Y)) if Y[i]==0] # class2 refers to data does not corresponds to flower Iris-Virginica  


# In[24]:


(mean_class1,std_class1)=gaussian_parameters(data_class1) # get each features gaussian parameters if their class is class1
(mean_class2,std_class2)=gaussian_parameters(data_class2) # get each features gaussian parameters if their class is class2
print(mean_class1,std_class1)
print(mean_class2,std_class2)
total_class1=0
for i in range(len(Y)):
    if(Y[i]==1):
        total_class1=total_class1+1
class1_probability=float(total_class1)/len(Y)
class2_probability=1-class1_probability


# In[25]:


predicted_labels=[]
for i in range(len(test_data)):
    probability_class1=1
    probability_class2=1
    for j in range(len(test_data[i])):
        probability_class1=probability_class1*probability(mean_class1[j],std_class1[j],test_data[i][j])
        probability_class2=probability_class2*probability(mean_class2[j],std_class2[j],test_data[i][j])
    probability_class1=probability_class1*class1_probability
    probability_class2=probability_class2*class2_probability
    if(probability_class1>probability_class2):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)


# # Accuracy score - Naive Bayes Classifier

# In[26]:


metrics.accuracy_score(test_labels,predicted_labels)


# # Precision - Naive Bayes Classifier

# In[27]:


metrics.precision_score(test_labels,predicted_labels)


# # F-measure - Naive Bayes Classifier

# In[28]:


metrics.f1_score(test_labels, predicted_labels)


# # Recall - Naive Bayes Classifier

# In[29]:


metrics.recall_score(test_labels,predicted_labels)


# # AUC - Naive Bayes Classifier

# In[30]:


fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
metrics.auc(fpr, tpr)


# # Logistic Regression - Gradient Descent

# Create a copy of features of test data and insert value "1" as first feature in every data point of test_data

# In[31]:


X_data=np.copy(X)
X_data=np.insert(X_data, 0, values=[1], axis=1)


# In[32]:


def sigmoid(z):
    return 1.0/(1+np.exp(-1*z))
def gradient_descent_logistic_regression(X_data,Y,learning_rate,number_iterations):
    theta=np.zeros(X_data.shape[1])
    for i in range(number_iterations):
        z=np.dot(X_data,theta)
        p=sigmoid(z)
        gradient=np.dot(X_data.T, (p - Y)) / Y.size
        theta=theta-learning_rate*gradient
    return theta


# In[33]:


learning_rate=0.1
number_iterations=30000
theta=gradient_descent_logistic_regression(X_data,Y,learning_rate,number_iterations)
print(theta)


# In[34]:


test_data_new=np.copy(test_data)
test_data_new=np.insert(test_data_new, 0, values=[1], axis=1);
predicted_labels=[]
for i in range(len(test_data_new)):
    if(sigmoid(np.dot(test_data_new[i],theta))>0.5):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)
# print(predicted_labels)


# # Accuracy score - Logistic Regression (Gradient Descent)

# In[35]:


metrics.accuracy_score(test_labels,predicted_labels)


# # Precision - Logistic Regression (Gradient Descent)

# In[36]:


metrics.precision_score(test_labels,predicted_labels)


# # F-measure - Logistic Regression (Gradient Descent)

# In[37]:


metrics.f1_score(test_labels, predicted_labels)


# # Recall - Logistic Regression (Gradient Descent)

# In[38]:


metrics.recall_score(test_labels,predicted_labels)


# # AUC - Logistic Regression (Gradient Descent)

# In[39]:


fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
metrics.auc(fpr, tpr)


# # Logistic Regression - Newton's method

# In[40]:


def newton_method_logistic_regression(X_data,Y,number_iterations):
    theta=np.zeros(X_data.shape[1])
    for i in range(number_iterations):
        z=np.dot(X_data,theta)
        p=sigmoid(z)
        gradient=np.dot(X_data.T, (p - Y)) / Y.size
        column=(np.ones(p.size)).T
        prob_product = np.dot(p,column-p)
        learning_rate=np.linalg.inv(np.dot(prob_product,np.dot(X_data.T,X_data)/ Y.size))
        theta=theta-np.dot(learning_rate,gradient)
    return theta


# In[41]:


theta=newton_method_logistic_regression(X_data,Y,number_iterations)
print(theta)


# In[42]:


predicted_labels=[]
for i in range(len(test_data_new)):
    if(sigmoid(np.dot(test_data_new[i],theta))>0.5):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)


# # Accuracy score - Logistic Regression (Newton's method)

# In[43]:


metrics.accuracy_score(test_labels,predicted_labels)


# # Precision - Logistic Regression (Newton's method)

# In[44]:


metrics.precision_score(test_labels,predicted_labels)


# # F-measure - Logistic Regression (Newton's method)

# In[45]:


metrics.f1_score(test_labels, predicted_labels)


# # Recall - Logistic Regression (Newton's method)

# In[46]:


metrics.recall_score(test_labels,predicted_labels)


# # AUC - Logistic Regression (Newton's method)

# In[47]:


fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
metrics.auc(fpr, tpr)


# # Logistic Regression (Library)

# In[48]:


logistic_regression=LogisticRegression()
logistic_regression.fit(X,Y)


# In[49]:


predicted_labels=logistic_regression.predict(test_data) # prediction of labels for test data


# # Accuracy score - Logistic Regression (library)

# In[50]:


metrics.accuracy_score(test_labels,predicted_labels)


# # Precision - Logistic Regression (library)

# In[51]:


metrics.precision_score(test_labels,predicted_labels)


# # F-measure - Logistic Regression (library)

# In[52]:


metrics.f1_score(test_labels, predicted_labels)


# # Recall - Logistic Regression (library)

# In[53]:


metrics.recall_score(test_labels,predicted_labels)


# # AUC - Logistic Regression (library)

# In[54]:


fpr, tpr, thresholds = metrics.roc_curve(test_labels, predicted_labels)
metrics.auc(fpr, tpr)

