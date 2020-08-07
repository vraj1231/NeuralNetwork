#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


# importing the data using pandas 
raw_data = pd.read_csv("Credit_Card_Applications.csv")
raw_data2 = raw_data.drop('Class', 1)
raw_data.head()


# In[3]:


#normalizing the data and seperating input and output data using iloc 
input_data = raw_data.iloc[:, :-1].values
output_data = raw_data.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))

scaled_input_data = sc.fit_transform(input_data)


# In[4]:


#traing the SOM model
from minisom import MiniSom

# 10x10 grid , input_len = 15 columns in input_data, default value, sigma = 1.0
som = MiniSom(x= 10, y = 10,input_len= 15, sigma = 1.0, learning_rate= 0.2)
som.random_weights_init(scaled_input_data)
som.train_random(scaled_input_data, num_iteration= 100)


# In[5]:


#visualizing the results
from pylab import bone, pcolor , colorbar, plot, show 

bone() # creates a white window which will create the map
pcolor(som.distance_map().T) #transforming distance map into different colors
colorbar() 
markers = ["o", "s"]
colors = ["r", "g"] 


for i, x in enumerate(scaled_input_data):
    winning_node = som.winner(x)
    plot(winning_node[0] + 0.5, # x-axis and 0.5 to put in center
         winning_node[1] + 0.5,
         markers[output_data[i]],
         markeredgecolor = colors[output_data[i]],
         markerfacecolor = "None",
         markersize = 10,
         markeredgewidth = 2)
show()
         
         
         


# In[6]:


#finding frauds in the bank 
Mapping = som.win_map(scaled_input_data)
frauds = np.concatenate((Mapping[(5,9)], Mapping [(5,2)]), axis= 0)

frauds = sc.inverse_transform(frauds) #inversing the normalizing value to original

frauds

df_frauds = pd.DataFrame(frauds, columns= raw_data2.columns)
df_frauds


# In[7]:


#creating ANN model 

customers = raw_data.iloc[:, 1:].values #we dont need customersid for ANN model
is_fraud = np.zeros(len(raw_data)) # using SOM frauds list makijng new vector

for i in range(len(raw_data)):
    if raw_data.iloc[i,0] in frauds:
        is_fraud[i] = 1

is_fraud


# In[8]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
customers = ss.fit_transform(customers)

input_layer = 15
hidden_layer = 2
outputsize = 1

model= tf.keras.Sequential([
    tf.keras.layers.Dense(input_layer),
    
    tf.keras.layers.Dense(hidden_layer, activation= "relu"),

    tf.keras.layers.Dense(outputsize, activation= "sigmoid"),
])

model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ['accuracy'])

# early_stopping = tf.keras.callbacks.EarlyStopping()

model.fit( customers, is_fraud,  batch_size = 1, epochs = 4, verbose= 2)


# In[9]:


#predicting the probablity of fraud

output_pred = model.predict(customers)

output_pred.shape


# In[10]:


output_pred = np.concatenate((raw_data.iloc[:, 0:1].values, output_pred), axis =1)

#output_pred = output_pred[output_pred[:,1].argsort()]

output_pred.shape


# In[11]:


output_pred = output_pred[output_pred[:,1].argsort()]

output_pred.shape


# In[12]:


df_output = pd.DataFrame(output_pred, columns= ["CustomerId", "Predictions"])

df_output


# In[ ]:




