
# coding: utf-8

# # Libraries 
# Libraries used in the following code are pandas, numpy, matplotlib, sklearn

# In[1]:


import pandas as pd
import numpy as np
import time
from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error


# # Data Read
# Reading data from csv file named as "housing.csv"

# In[2]:


data=pd.read_csv("housing.csv")
data.head()


# In[3]:


data.describe()


# # Plotting Data
# Plotting each attribute/feature in the data set

# In[4]:


data.hist(bins=50,figsize=(20,15))


# # Creating Training and Test Dataset
# creating a training and test dataset using Scikit-Learn

# In[5]:


train_set, test_set=train_test_split(data,test_size=0.2,random_state=12)


# # Correlations of Data withrespect to median_house_value 
# Correlations of data withrespect to median_house_value to be estimated
# 

# In[6]:


correlation_matrix=train_set.corr()
correlation_matrix["median_house_value"].sort_values(ascending=False)


# # Attribute Combinations
# Constructing new attributes using already existing attributes

# In[7]:


train_set1=train_set.copy()
train_set1["rooms_per_household"] =train_set1["total_rooms"]/train_set1["households"]
train_set1["bedrooms_per_room"] = train_set1["total_bedrooms"]/train_set1["total_rooms"]
train_set1["population_per_household"]=train_set1["population"]/train_set1["households"]
train_set=train_set1


# In[8]:


correlation_matrix=train_set.corr()
correlation_matrix["median_house_value"].sort_values(ascending=False)


# # Preparing the Data
# Here we will first remove the "median_house_value" entries from "train_set"

# In[9]:


data_new = train_set.drop("median_house_value", axis=1)
data_labels = train_set["median_house_value"].copy()
train_labels = data_labels.copy()
data_new.head()


# In[10]:


data_test_new = test_set.drop("median_house_value", axis=1)
data_test_labels = test_set["median_house_value"].copy()
test_labels = data_test_labels.copy()
data_test_new.head()


# In[11]:


data_labels.head()


# In[12]:


sample_incomplete_rows = data_new[data_new.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[13]:


sample_test_incomplete_rows = test_set[test_set.isnull().any(axis=1)].head()
sample_test_incomplete_rows


# In[14]:


sample_incomplete_rows.dropna(subset=["total_bedrooms"]) # drop the row corresponding to "total_bedrooms" NaN
sample_test_incomplete_rows.dropna(subset=["total_bedrooms"]) 


# In[15]:


sample_incomplete_rows.drop("total_bedrooms", axis=1)       # drop the entire column "total_bedrooms"
sample_test_incomplete_rows.drop("total_bedrooms", axis=1) 


# # Filling missing Data
# Filling missing data with median value

# In[16]:


median = data_new["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
print(sample_incomplete_rows.head())


# In[17]:


test_median = data_test_new["total_bedrooms"].median()
sample_test_incomplete_rows["total_bedrooms"].fillna(test_median, inplace=True)
print(sample_test_incomplete_rows.head())


# In[18]:


imputer = Imputer(strategy="median")
data_num = data_new.drop('ocean_proximity', axis=1)
data_num["rooms_per_household"] =data_num["total_rooms"]/data_num["households"]
data_num["bedrooms_per_room"] = data_num["total_bedrooms"]/data_num["total_rooms"]
data_num["population_per_household"]=data_num["population"]/data_num["households"]
imputer.fit(data_num)
data_num.median().values


# In[19]:


imputer_test = Imputer(strategy="median")
data_test_num = data_test_new.drop('ocean_proximity', axis=1)
data_test_num["rooms_per_household"] =data_test_num["total_rooms"]/data_test_num["households"]
data_test_num["bedrooms_per_room"] = data_test_num["total_bedrooms"]/data_test_num["total_rooms"]
data_test_num["population_per_household"]=data_test_num["population"]/data_test_num["households"]
imputer_test.fit(data_test_num)
data_test_num.median().values


# In[20]:


X = imputer.transform(data_num)
housing_tr = pd.DataFrame(X, columns=data_num.columns, index = list(data_num.index.values))
housing_tr.head()


# In[21]:


X_test = imputer_test.transform(data_test_num)
housing_te = pd.DataFrame(X_test, columns=data_test_num.columns, index = list(data_test_num.index.values))
housing_te.head()


# # Handling Text attributes
# Here we will use one hot encoding to handle text attributes

# In[22]:


housing_category = data_new['ocean_proximity']
housing_category.head(10)


# In[23]:


housing_test_category = data_test_new['ocean_proximity']
housing_test_category.head(10)


# In[24]:


housing_category.value_counts()


# In[25]:


housing_test_category.value_counts()


# In[26]:


housing_category_encoded, housing_categories = housing_category.factorize()
housing_category_encoded[:10]


# In[27]:


housing_test_category_encoded, housing_test_categories = housing_test_category.factorize()
housing_test_category_encoded[:10]


# In[28]:


housing_categories


# In[29]:


housing_test_categories


# In[30]:


encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_category_encoded.reshape(-1,1))
housing_cat_1hot


# In[31]:


encoder_test = OneHotEncoder()
housing_test_cat_1hot = encoder_test.fit_transform(housing_test_category_encoded.reshape(-1,1))
housing_test_cat_1hot


# In[32]:


encoder = ce.OneHotEncoder()
housing_cat_reshaped = housing_category.values.reshape(-1, 1)
encoder.fit(housing_cat_reshaped)
X_cleaned = encoder.transform(housing_cat_reshaped)
cat_data = X_cleaned.as_matrix()
print(X_cleaned[0:5])
print(type(cat_data))
print(cat_data.shape)


# In[33]:


encoder_test = ce.OneHotEncoder()
housing_test_cat_reshaped = housing_test_category.values.reshape(-1, 1)
encoder_test.fit(housing_test_cat_reshaped)
X_test_cleaned = encoder_test.transform(housing_test_cat_reshaped)
cat_test_data = X_test_cleaned.as_matrix()
print(X_test_cleaned[0:5])
print(type(cat_test_data))
print(cat_test_data.shape)


# # Scaling

# In[34]:


scaler = StandardScaler()
scaler.fit(housing_tr)
housing_data = scaler.transform(housing_tr)
type(housing_data[:5])
housing_data[:5]


# In[35]:


scaler_test = StandardScaler()
scaler.fit(housing_te)
housing_test_data = scaler.transform(housing_te)
type(housing_test_data[:5])
housing_test_data[:5]


# # Linear Regression - Closed Form

# In[36]:


def normal_equation(x,y):
    z1 = np.dot(x.transpose(), x) 
    z = np.linalg.inv(z1)
    z2 = np.dot(z, x.transpose())
    theta = np.dot(z2, y)
    return theta


# In[37]:


X = np.array(housing_data)
Y = np.array(train_labels).flatten()
X=np.insert(X, 0, values=1, axis=1)


# ## Training time - Linear Regression (Closed form)

# In[38]:


start_time = time.time()
theta=normal_equation(X, Y)
end_time = time.time()
training_time=end_time-start_time
print(training_time)


# In[39]:


print(theta)


# In[40]:


X_test = np.array(housing_test_data)
Y_test = np.array(test_labels).flatten()
X_test=np.insert(X_test, 0, values=1, axis=1)


# ## Checking Model Fitting - Linear Regression - Closed Form

# In[41]:


predicted_labels=[]
for i in range(Y.size):
    predicted_labels.append(X[i].dot(theta))


# In[42]:


predicted_test_labels=[]
for i in range(Y_test.size):
    predicted_test_labels.append(X_test[i].dot(theta))


# ## Mean Square Error - Linear Regression - Closed Form

# In[43]:


mean_squared_error(train_labels, predicted_labels)


# In[44]:


mean_squared_error(test_labels, predicted_test_labels)


# # Linear Regression- Gradient Desent

# In[45]:


def gradient_descent_linear_regression(X_data,Y,learning_rate,number_iterations):
    theta=np.zeros(X_data.shape[1])
    for i in range(number_iterations):
        z=np.dot(X_data,theta)
        gradient=np.dot(X_data.T, (z - Y)) / Y.size
        theta=theta-learning_rate*gradient
    return theta


# ## Training Time - Linear Regression (Newton's Method)

# In[46]:


learning_rate=0.1
number_iterations=30000
start_time = time.time()
theta=gradient_descent_linear_regression(X,Y,learning_rate,number_iterations)
end_time = time.time()
training_time=end_time-start_time
print(training_time)


# In[47]:


predicted_labels=[]
for i in range(len(Y)):
        predicted_labels.append(X[i].dot(theta))


# In[48]:


predicted_test_labels=[]
for i in range(len(Y_test)):
        predicted_test_labels.append(X_test[i].dot(theta))


# ## Mean Square Error - Linear Regression (Gradient Desent)

# In[49]:


mean_squared_error(train_labels,predicted_labels)


# In[50]:


mean_squared_error(test_labels,predicted_test_labels)


# # Linear Regression - Newton's method

# In[51]:


def newton_method_linear_regression(X_data,Y,number_iterations):
    theta=np.zeros(X_data.shape[1])
    for i in range(number_iterations):
        z=np.dot(X_data,theta)
        gradient=np.dot(X_data.T, (z - Y)) / Y.size
        learning_rate=np.linalg.inv(np.dot(X_data.T,X_data)/ Y.size)
        theta=theta-np.dot(learning_rate,gradient)
    return theta


# ## Training Time - Linear Regression (Newton's Method)

# In[52]:


start_time = time.time()
theta=newton_method_linear_regression(X,Y,number_iterations)
end_time = time.time()
training_time=end_time-start_time
print(training_time)


# In[53]:


predicted_labels=[]
for i in range(len(Y)):
    predicted_labels.append(X[i].dot(theta))


# In[54]:


predicted_test_labels=[]
for i in range(len(Y_test)):
    predicted_test_labels.append(X_test[i].dot(theta))


# ## Mean Square Error - Linear Regression (Newton's method)

# In[55]:


mean_squared_error(train_labels,predicted_labels)


# In[56]:


mean_squared_error(test_labels,predicted_test_labels)


# # Ridge Regression 

# In[57]:


def ridge_regression(x,y,regularisation_parameter):
    z1 = np.dot(x.transpose(), x)
    a = np.zeros((x.shape[1], x.shape[1]))
    np.fill_diagonal(a, regularisation_parameter)
    s=np.add(z1, a)
    z = np.linalg.inv(s)
    z2 = np.dot(z, x.transpose())
    theta = np.dot(z2, y)
    return theta


# ## Training Time - Ridge Regression 

# In[58]:


start_time = time.time()
alphas = 10**np.linspace(10,-2,100)*0.5
alpha_min=alphas[0]
theta_min=ridge_regression(X, Y,alpha_min)
predicted_labels=[]
for i in range(Y.size):
    predicted_labels.append(X[i].dot(theta_min))
mse_min=mean_squared_error(train_labels, predicted_labels)

for a in range(1,len(alphas)):
    theta=ridge_regression(X, Y,alphas[a])
    predicted_labels=[]
    for i in range(Y.size):
        predicted_labels.append(X[i].dot(theta))
    mse=mean_squared_error(train_labels, predicted_labels)
    if(mse < mse_min):
        alpha_min=alphas[a]
        theta_min=theta
        mse_min=mse 
    
    
end_time = time.time()
training_time=end_time-start_time
print(training_time)


# In[59]:


alpha_min          #regularization parameter for which mse in minium


# In[60]:


predicted_labels=[]
for i in range(Y.size):
    predicted_labels.append(X[i].dot(theta_min))


# In[61]:


predicted_test_labels=[]
for i in range(Y_test.size):
    predicted_test_labels.append(X_test[i].dot(theta_min))


# ## Mean Square Error  - Ridge Regression

# In[62]:


mean_squared_error(train_labels, predicted_labels)


# In[63]:


mean_squared_error(test_labels, predicted_test_labels)


# # Lasso Regression

# ## Training Time - Lasso Regression

# In[64]:


# Lasso Regression

alphas = 10**np.linspace(10,-2,100)*0.5
start_time = time.time()
lasso = Lasso(max_iter = 10000, normalize = True)
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X, Y)
    coefs.append(lasso.coef_)
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X, Y)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X, Y)
end_time = time.time()
training_time=end_time-start_time
print(training_time)



# ## Mean Square Error - Lasso Regression

# In[65]:


mean_squared_error(Y, lasso.predict(X))


# In[66]:


mean_squared_error(Y_test, lasso.predict(X_test))

