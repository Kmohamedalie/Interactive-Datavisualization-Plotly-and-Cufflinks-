#!/usr/bin/env python
# coding: utf-8

# # Plotly and Cufflinks  

#  <h3>Simple Introduction</h3>a gentle introduction of:

# - <a href="https://www.plotly.com/">Plotly</a> is a free and open-source graphing library for Python.

# - <a href="https://plotly.com/python/plotly-express/">Plotly Express</a>

# - <a href="https://pypi.org/project/cufflinks/">Cufflinks</a> his <a href="https://github.com/santosjorge/cufflinks">library</a> binds the power of plotly with the flexibility of pandas for easy plotting.

# #### Install  Plotly and Cufflinks

# <code>pip install plotly</code>

# <code>pip install cufflinks</code>

# In[73]:


# import libraries
import pandas as pd
import numpy as np
import cufflinks as cf


# Ploty can host your visualization online and offline.

# In[74]:


# import plotly offline
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[75]:


# notebook to work 
init_notebook_mode(connected = True)


# In[76]:


# to use cufflinks offline
cf.go_offline()


# In[77]:


# create a dataframe with 100 rows and 4 columns for exploratory purpose
df = pd.DataFrame(np.random.randn(100,4),columns="A B C D".split())
df.head()


# In[78]:


df2 = pd.DataFrame({'Category':['A', 'B', 'C']
                    ,'Values':[32,43,50]})
df2.head()


# ## Line:

# Line plot for timeseries

# In[79]:


# to make an interactive plot plotly
df.iplot(title="Interactive random plot");


# ## Scatter:

# Scatter for relationship

# In[80]:


# scatter
df.iplot(kind='scatter',x='A',y='B',mode='markers', title="Scatter")


# ## Bar:

# Comparison across different subgroups of your data.

# In[81]:


# to make an interactive plot plotly
df2.iplot(kind='bar', x='Category', y='Values', title="Bar");


# Aggregation function:

# In[82]:


# to make an interactive plot plotly
df.sum().iplot(kind='bar', title="Aggregate Barplot");


# ## Boxplot

# Distribution of dataset

# In[83]:


# to make an interactive plot plotly
df.iplot(kind='box', title="Boxplot");


# ## Surface Plot:

# In[84]:


df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[5,4,3,2,1]})
df3.head()


# In[85]:


# 3D surface plot
df3.iplot(kind='surface', colorscale='rdylbu')


# ## Histogram Plot:

# Distribution of continous variable.

# In[86]:


# click on the legend to show a specific category
df.iplot(kind='hist')


# ## Spread:

# In[87]:


df[['A','B']].iplot(kind='spread');


# ## Bubble chart:

# In[112]:


df.iplot(kind='bubble', x='A', y='B', size='C')


# ## Scatter_matrix:

# In[116]:


df.scatter_matrix()
# not for very lage dataset the kernel might crash


# # Plotly express

# #### Machine Learning 

# In[66]:


import plotly.express as px
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, random_state=0)

model = LogisticRegression()
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_score)

# The histogram of scores compared to true labels
fig_hist = px.histogram(
    x=y_score, color=y, nbins=50,
    labels=dict(color='True Labels', x='Score')
)

fig_hist.show()


# Evaluating model performance at various thresholds
df = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr
}, index=thresholds)
df.index.name = "Thresholds"
df.columns.name = "Rate"

fig_thresh = px.line(
    df, title='TPR and FPR at every threshold',
    width=700, height=500
)

fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
fig_thresh.show()


# #### Maps:

# In[67]:


# using gapminder data
import plotly.express as px
df = px.data.gapminder()
fig = px.scatter_geo(df, locations="iso_alpha", color="continent", hover_name="country", size="pop",
               animation_frame="year", projection="natural earth")
fig.show()


# In[69]:


import plotly.express as px
df = px.data.gapminder()
fig = px.choropleth(df, locations="iso_alpha", color="lifeExp", hover_name="country", animation_frame="year", range_color=[20,80])
fig.show()

