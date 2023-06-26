#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Importing packages to use

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#Loading the datasets
results = pd.read_csv(r'C:\Users\aturk\Downloads\results.csv', header=0)
races=pd.read_csv(r'C:\Users\aturk\Downloads\races.csv', header=0)
drivers=pd.read_csv(r'C:\Users\aturk\Downloads\drivers.csv', header=0)
constructors=pd.read_csv(r'C:\Users\aturk\Downloads\constructors.csv', header=0)


# In[5]:


results.info()


# In[6]:


races.info()


# In[7]:


drivers.info()


# In[8]:


constructors.info()


# In[9]:


#Left Joint of results and races datasets, merging results dataset with specific columns of races datasets 
df= pd.merge(results, races[["raceId","year","name","round"]], on="raceId", how="left")


# In[10]:


df.info()


# In[11]:


#Now merging driver and constructor datasets with df
df=pd.merge(df, drivers[["driverId","driverRef","nationality"]], on="driverId", how="left")
df=pd.merge(df, constructors[["constructorId","name","nationality"]], on="constructorId", how="left")


# In[12]:


df.info()


# In[13]:


df.head()


# In[14]:


#Dropping unnecessary columns
df.drop(["number", 'position','positionText','laps','fastestLap','statusId','resultId','raceId','driverId','constructorId'], axis=1, inplace= True)


# In[15]:


df.head()


# In[16]:


#Renaming some columns
df.rename(columns={'driverRef':'DriverName','rank':'fastest_lap_rank','name_x':'gp_place','nationality_x':'driver_nationality','name_y':'constructor_name','nationality_y':'constructor_nationality','driver_ref':'driver','positionOrder':'Finishing_position','grid':'Starting_position'}, inplace=True)


# In[17]:


df.head()


# In[18]:


print(df['year'].unique())


# In[19]:


#2019 data is incomplete, so drop 2019 data
df=df[df['year']!=2019]
print(df['year'].unique())


# In[20]:


#Sort values based on year, round and position order
df=df.sort_values(by=["year",'round','Finishing_position'], ascending=[False, True, True])


# In[21]:


#we see that if a driver didn't finish a race, their time is showsn as \N
df.tail(10)


# In[22]:


#Replacing \N values with proper NaN
df.time.replace('\\N',np.nan, inplace=True)
df.milliseconds.replace('\\N',np.nan, inplace=True)
df.fastest_lap_rank.replace('\\N',np.nan, inplace=True)
df.fastestLapTime.replace('\\N',np.nan, inplace=True)
df.fastestLapSpeed.replace('\\N',np.nan, inplace=True)


# In[23]:


df.tail(10)


# In[24]:


#Changing the datatypes of NaNs
df.milliseconds=df.milliseconds.astype(float)
df.fastest_lap_rank=df.fastest_lap_rank.astype(float)
df.fastestLapSpeed=df.fastestLapSpeed.astype(float)


# In[25]:


#resetting the index after changes
df.reset_index(drop=True, inplace=True)
df.head()


# In[26]:


#Shape
print(df.shape)


# In[27]:


df.info()


# In[28]:


#Setting size and color for visuals
sns.set_palette('Set3')
plt.rcParams['figure.figsize']=10,6


# In[29]:


#Creating a GP Winners dataframe
driver_winner=df.loc[df['Finishing_position']==1].groupby('DriverName')['Finishing_position'].count().sort_values(ascending=False).to_frame().reset_index()


# In[30]:


#Bar Plot
sns.barplot(data=driver_winner, y='DriverName', x='Finishing_position', color='green', alpha=0.8)
plt.title('Most GP Winners in F1')
plt.ylabel('Driver Name')
plt.xlabel('Number of GP Wins')
plt.yticks([]) #avoid crowded/unreadable y labels


# In[31]:


#Bar Plot for winners in the last 3 years
#Creating a GP Winners dataframe
driver_winner_recent=df.loc[(df['Finishing_position']==1) & (df['year']>2019)].groupby('DriverName')['Finishing_position'].count().sort_values(ascending=False).to_frame().reset_index()
sns.barplot(data=driver_winner_recent, y='DriverName', x='Finishing_position', color='green', alpha=0.8)
plt.title('Most GP Winners in F1 since 2020')
plt.ylabel('Driver Name')
plt.xlabel('Number of GP Wins')


# In[32]:


#Create a new dataframe for top 10 drivers
top10Drivers=driver_winner.head(10)
print(top10Drivers)


# In[33]:


#top 10 drivers plot
sns.barplot(data=top10Drivers, y='DriverName', x='Finishing_position', color='blue', alpha=0.7, linewidth=0.8, edgecolor='black')
plt.title('Most GP Winners in F1')
plt.ylabel('Driver Name')
plt.xlabel('Number of GP Wins')


# In[34]:


#top constructors
cons_winner=df.loc[df['Finishing_position']==1].groupby('constructor_name')['Finishing_position'].count().sort_values(ascending=False).to_frame().reset_index()
#Bar Plot
sns.barplot(data=cons_winner, y='constructor_name', x='Finishing_position', color='red', alpha=0.8)
plt.title('Most GP Winners in F1')
plt.ylabel('Constructor')
plt.xlabel('Number of GP Wins')
plt.yticks([]) #avoid crowded/unreadable y labels


# In[35]:


#top 10 winner constructors
top10Cons=cons_winner.head(10)
print(top10Cons)


# In[36]:


#bar plot of top 10 constructors
sns.barplot(data=top10Cons, x='Finishing_position', y='constructor_name', color='pink', alpha=0.9, edgecolor='black')
plt.title('Ferrari dominated other constructors')
plt.xlabel('Number of GP Wins')
plt.ylabel('Constructor')
#plt.yticks([]) #this deletes y values


# In[37]:


#0 start position means driver started from pit lane
df_no_zero=df[df['Starting_position'] !=0]

#regression plot to display if 
plt.figure(figsize = [12,7])
sns.regplot(data=df_no_zero, x='Starting_position', y='Finishing_position', x_jitter=0.3, y_jitter=0.3, scatter_kws={'alpha':1/5})
plt.title('Starting Position vs Finishing Position')
plt.xlabel('Starting Position')
plt.ylabel('Finishing Position')


# In[38]:


#To answer Andrew's question,when is Mclaren's first win
print(df_no_zero.loc[(df_no_zero['constructor_name']=='McLaren') & (df_no_zero['Finishing_position']==1)]['year'].min())


# In[43]:


#Speed data is available since 2004, creating a new dataframe to display if speed of cars changed
df_speed=df[df['year']>=2004]
#to_frame() converts dataset to dataframe
df_group_speed=df_speed.groupby(['gp_place','year'])['fastestLapSpeed'].mean().to_frame().reset_index()

#creating facetgrid
fgrid=sns.FacetGrid(data=df_group_speed, col = 'gp_place', col_wrap=5)
fgrid.map(plt.scatter, 'year','fastestLapSpeed', alpha=0.8, linewidth=0.8, edgecolor='red', s=100)
fgrid.set_titles("{col_name}")
fgrid.set_xlabels('Year')
fgrid.set_ylabels('Average fastest speed (km/h)')
plt.subplots_adjust(top=0.92)
fgrid.fig.suptitle('Average of Fastest Lap Speed of All Teams Since 2004')


# In[40]:


df_group_speed.head()


# In[42]:


df_italy=df_group_speed[df_group_speed['gp_place']=='Italian Grand Prix']
#creating facetgrid
fgrid=sns.FacetGrid(data=df_italy, col = 'gp_place', col_wrap=5)
fgrid.map(plt.scatter, 'year','fastestLapSpeed', alpha=0.8, linewidth=0.8, edgecolor='red', s=100)
fgrid.set_titles("{col_name}")
fgrid.set_xlabels('Year')
fgrid.set_ylabels('Average fastest speed (km/h)')
plt.subplots_adjust(top=0.92)
fgrid.fig.suptitle('Average of Fastest Lap Speed of All Teams Since 2004 in Italian GP')


# In[ ]:




