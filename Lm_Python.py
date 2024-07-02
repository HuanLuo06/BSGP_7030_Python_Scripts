#!/usr/bin/env python
# coding: utf-8

# # BSGP 7030 First Application Assignment

# ## Python Version

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import sys

print("\nBSGP 7030 Python LM Example\n")

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    print("Please provide an input file")
    sys.exit(-1)

print(f"loading{input_file}")
print()
df = pd.read_csv(input_file)

print("File loaded looking at the head of the file")
print(df.head())

print("plotting data\n")

# In[8]:

plt.scatter(df['x'], df['y'])
plt.savefig("orginal_data.png")

print("Modeling data\n")

# In[9]:

model = LinearRegression()

x = np.array(df['x']).reshape(-1,1)
y = np.array(df['y'])
model.fit(x,y)

r_sq = model.score(x,y)

print("Model summary\n")
print(f"intercept:{model.intercept_}")
print(f"slope:{model.coef_}")
print(f"slope:{r_sq}")
print()

print("Predicting data\n")
y_pred = model.predict(x)

# In[ ]:
print("plotting fit\n")
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.savefig("fit_data.png")


