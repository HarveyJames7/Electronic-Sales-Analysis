#!/usr/bin/env python
# coding: utf-8

# In[1]:


# I will collect, clean, transform, and analyze the dataset from Kaggle based on eletronic stores sales
# I will use this to help answer the following business related questions

# Question 1: What was the best month for sales and what was earned that month?
# Question 2: What city had the highest sales and what were their sale numbers?
# Question 3: What time should we display advertisments to maximize likelihood of customer's buying product?
# Question 4: What products are most often sold together in pairs?
# Question 5: What item sold the most and why?

import pandas as pd
import os


# In[10]:


# Merge 12 months of sales data into 1 file

df = pd.read_csv("./Sales_Data/Sales_April_2019.csv")

files = [file for file in os.listdir('./Sales_Data')]

all_months_data = pd.DataFrame()
for file in files:
    df = pd.read_csv("./Sales_Data/" + file)
    all_months_data = pd.concat([all_months_data, df])
all_months_data.to_csv("all_data.csv", index = False)
all_data = pd.read_csv("all_data.csv")


# In[16]:


# Augment data with additional colums
all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype("int32")
all_data.head()


# In[25]:


# Remove NaN's
nan_df = all_data[all_data.isna().any(axis=1)]
all_data = all_data.dropna(how = "all")
all_data['Month'] = all_data['Month'].astype("int32")

# Remove Or
all_data = all_data[all_data["Order Date"].str[0:2] != "Or"]
all_data.head()


# In[40]:


# Add Sales Column
all_data["Quantity Ordered"] = pd.to_numeric(all_data["Quantity Ordered"])
all_data["Price Each"] = pd.to_numeric(all_data["Price Each"])
all_data["Sales"] = all_data["Quantity Ordered"] * all_data["Price Each"]
all_data.head()


# In[39]:


# Reorder Colums
all_data = all_data[[
 'Order ID',
 'Product',
 'Quantity Ordered',
 'Price Each',
 'Sales',
 'Order Date',
 'Month',
 'Purchase Address']]
cols = list(all_data.columns.values)
cols


# In[76]:


# Add a city & state column (seperate and also combined)
def get_city(address):
    return address.split(',')[1]
def get_state(address):
    return address.split(',')[2].split (' ')[1]
all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")
all_data.head()

# Different way to create columns
        # all_data['City'] = all_data['Purchase Address'].apply(lambda x: x.split(',')[1])
        # all_data['State'] = all_data['Purchase Address'].apply(lambda x: x.split(',')[2].split(' ')[1])


# In[65]:


# Question: 1 What was the best month for sales and what was earned that month?
# Answer: December: $461,3443
pd.options.display.float_format = '{:.0f}'.format

results = all_data.groupby('Month').sum()
print(results)


# In[63]:


# Graph for #1
import matplotlib.pyplot as plt
import numpy as np
plt.ticklabel_format(style='plain')

months = range(1,13)
plt.bar(months, results['Sales'])
plt.xticks(months)
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month Number')
plt.show()


# In[77]:


# Question 2: What city had the highest sales and what were their sale numbers?
# Answer: San Francisco: $826,2204

results2 = all_data.groupby('City').sum()
results2


# In[81]:


# Graph for #2
plt.ticklabel_format(style='plain')

cities = [city for city, df in all_data.groupby('City')]
plt.bar(cities, results2['Sales'])
plt.xticks(cities, rotation = 'vertical', size = 9)
plt.ylabel('Sales in USD ($)')
plt.xlabel('City Name')
plt.show()


# In[90]:


# Question 3: What time should we display advertisments to maximize likelihood of customer's buying product?

all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
all_data['Hour'] = all_data['Order Date'].dt.hour
all_data['Minute'] = all_data['Order Date'].dt.minute
all_data.head()
hours = [hour for hour, df in all_data.groupby('Hour')]
plt.plot(hours, all_data.groupby(['Hour']).count())
plt.xticks(hours)
plt.xlabel('Hour')
plt.ylabel("Number of Orders")
plt.grid()
plt.show()
# Question 3 Answer
# A good time to display an ad would be 9am-12am and 6pm-8pm because this is when most orders are placed


# In[100]:


# Question 4: What products are most often sold together in pairs?
from itertools import combinations
from collections import Counter

df = all_data[all_data["Order ID"].duplicated(keep = False)]
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))
df = df[['Order ID', 'Grouped']].drop_duplicates()

count = Counter()
for row in df['Grouped']:
    row_list = row.split(',')
    count.update(Counter(combinations(row_list, 2)))
for key, value in count.most_common(10):
    print(key, value)
# Question 4 Answer: Here is a list of the top 10 items that are sold together in pairs


# In[116]:


# Question 5: What item sold the most and why?
product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']

products = [product for product, df in product_group]
plt.bar(products, quantity_ordered)
plt.ylabel('Quantity Ordered')
plt.xlabel("Product")
plt.xticks(products, rotation = 'vertical', size = 9)
plt.show()
# Question 5 Answer: AAA Batteries are the most sold items. 
# The reason for this is most likely due to the cost of the batteries being low


# In[122]:


prices = all_data.groupby('Product').mean()['Price Each']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(products, quantity_ordered, color = 'r')
ax2.plot(products, prices, 'b-')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color = 'r')
ax2.set_ylabel('Price ($)', color = 'b')
ax1.set_xticklabels(products, rotation = 'vertical', size = 9)

plt.show()
# This chart shows the relationship between the price and quantity ordered. 
# This shows why AAA batteries were ordered the most. They were one of the lowest prices.

