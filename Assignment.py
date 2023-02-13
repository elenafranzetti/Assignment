#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#import os
import os
#directory
os.chdir('C:\\Users\\marty\\Desktop\\assignment')
#check
os.getcwd()


# In[4]:


#open dataset
df = pd.read_csv('trans.csv', sep=";")


# In[5]:


# show a preview
df.head()


# In[6]:


# check shape
df.shape


# In[7]:


#check column list
df.columns


# In[8]:


from datetime import datetime


# In[9]:


df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")


# In[10]:


df['data']


# In[11]:


df["mese"] = df["data"].dt.month


# In[12]:


#to check
df['mese']


# In[13]:


#creare la nuova variabile trimestre
mesi = [1,2,3,4,5,6,7,8,9,10,11,12]
trimestre = ["Trim1", "Trim1", "Trim1", "Trim2", "Trim2", "Trim2", "Trim3", "Trim3", "Trim3", "Trim4", "Trim4", "Trim4"]


# In[14]:


#nuova colonna
df['Trimestre'] = df['mese'].replace(mesi, trimestre)


# In[15]:


#to check
df.head()


# In[18]:


#bilancio annuale entrate meno uscite
df['amount'].sum()


# In[19]:


#salvare
bilancio_annuale = 7525.14


# In[20]:


bilancio_annuale


# In[21]:


#bilancio diviso per mese
bilancio_mensile = df.groupby('mese')['amount'].sum()


# In[22]:


bilancio_mensile


# In[23]:


bilancio_trim = df.groupby('Trimestre')['amount'].sum()


# In[24]:


bilancio_trim


# In[25]:


# x-coordinates of bars 
base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


# In[26]:


# labels for bars
tick_label = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']


# In[27]:


#Visualizzazione bilancio mensile

# x-coordinates of left sides of bars 
base = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  
# labels for bars
tick_label = ['Gennaio', 'Febbraio', 'Marzo', 'Aprile', 'Maggio', 'Giugno', 'Luglio', 'Agosto',
             'Settembre', 'Ottobre', 'Novembre', 'Dicembre']
#Color
color = ['green', 'red', 'red', 'red', 'green', 'red', 'red', 'red', 'red', 'green', 'red', 'green']


# In[28]:


import seaborn as sns


# In[29]:


# plotting a bar chart
# Figure Size
fig, ax = plt.subplots(figsize =(13, 8))
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
plt.bar(base, bilancio_mensile, tick_label = tick_label,
        width = 0.8, color = color) 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)

# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 

 # Add x, y gridlines
sns.set_style("whitegrid")
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
# Add Plot Title
ax.set_title('Bilancio Mensile (Euro)',
             loc ='left', )
# Show Plot
plt.show()


# In[30]:


#bilancio trimestrale 
categories = ["Primo Trimestre", "Secondo Trimestre", "Terzo Trimestre", "Quarto Trimestre"]
counts = [3322, 5049, -5046, 4200]
trim = pd.DataFrame(list(zip(categories, counts)), columns =['categories', 'counts'])


# In[31]:


#Visualizzazione bilancio trimestrale

fig, ax = plt.subplots(figsize =(10, 5))

# Show Plot 
 # Add x, y gridlines
sns.set_style("whitegrid")
sns.barplot(x="categories", y="counts", palette='Greens', data=trim)
plt.xlabel('')
plt.ylabel('')
plt.title('Bilancio Trimestrale (Euro)', size=18, color='grey')
plt.xticks(size=14, color='grey')
plt.yticks(size=14, color='grey')
sns.despine(left=True)


# In[33]:


#distribuzione di frequenza
get_ipython().run_line_magic('matplotlib', 'inline')

category_count = df['categoria'].value_counts()
sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize =(13, 8))
sns.barplot(category_count.index, category_count.values, alpha=0.9, data=df)
plt.xticks(rotation = 45, ha="right", size=10)
plt.title('Distribuzione di Frequenza delle Categorie di Transazione')
plt.ylabel('')
plt.xlabel('Tipo di Transazione', fontsize=12)
plt.show()


# In[34]:


#grafico a torta
labels = df['categoria'].astype('category').cat.categories.tolist()
counts = df['categoria'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%') #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# In[35]:


#creare un oggetto che conta quante volte una categoria compare in valori assoluti
counts = df['categoria'].value_counts()
counts


# In[36]:


#Bar plot con percentuali sulla y
# Example data

# Calculate the percentage of each value
data_perc = [i/sum(counts) for i in counts]

# Plot the bar plot with percentages

sns.set(style="darkgrid")
fig, ax = plt.subplots(figsize =(13, 8))
# Add the percentage labels to each bar

for i, v in enumerate(data_perc):
    ax.text(i, v + 0.01, str(round(v*100, 2)) + "%", color='black')

sns.barplot(category_count.index, data_perc, alpha=0.9, data=df)
plt.xticks(rotation = 45, ha="right", size=10, fontweight='bold')
plt.title('Distribuzione di Frequenza delle Categorie di Transazione (%)', fontweight='bold')
plt.ylabel('')
plt.xlabel('Tipo di Transazione', fontsize=12)
plt.show()


# In[37]:


#valori assoluti
count_trim = df.groupby('Trimestre')['categoria'].value_counts()
count_trim


# In[38]:


#percentuale
count_trim = df.groupby('Trimestre')['categoria'].value_counts(normalize=True)
count_trim_perc = count_trim * 100
count_trim_perc


# In[38]:


cat_trim = [[1, 2, 3, 4], [38.5, 33.3, 34.5, 36.7], [27, 33.3, 34.5, 36.7], [11.5, 10, 10.3, 10], [11.5, 10, 3.4, 0], [11.5, 10, 10.3, 7], [0, 3.3, 0, 0], [0, 0, 7, 10]]
etichette = ['Incasso Fattura', 'Pagamento fornitori', 'Romborso Finanziamento', 'Stipendi', 'Utenze', 'Erogazione Finanziamento', 'Pagamento Fattura']


# In[40]:


g1 = [38.5, 33.3, 34.5, 36.7]
g2 = [27, 33.3, 34.5, 36.7]
g3 = [11.5, 10, 10.3, 10]
g4 = [11.5, 10, 3.4, 0]
g5 = [11.5, 10, 10.3, 7]
g6 = [0, 3.3, 0, 0]
g7 = [0, 0, 7, 10]
# Combine the data into one array
cat_trim = np.array([g1, g2, g3, g4, g5, g6, g7])


# In[41]:


# Plot the data as a stacked bar plot
# Generate data for the bar plot
fig, ax = plt.subplots(figsize =(13, 8))
ax.bar(np.arange(4), cat_trim[0,:], label = 'Incasso Fattura')
ax.bar(np.arange(4), cat_trim[1,:], bottom=cat_trim[0,:], label = 'Pagamento fornitori')
ax.bar(np.arange(4), cat_trim[2,:], bottom=cat_trim[0,:] + cat_trim[1, :], label = 'Rimborso finanziamento')
ax.bar(np.arange(4), cat_trim[3,:], bottom=cat_trim[0,:] + cat_trim[1, :] + cat_trim[2, :], label = 'Stipendi')
ax.bar(np.arange(4), cat_trim[4,:], bottom=cat_trim[0,:] + cat_trim[1, :] + cat_trim[2, :] + cat_trim[3, :], label = 'Utenze')
ax.bar(np.arange(4), cat_trim[5,:], bottom=cat_trim[0,:] + cat_trim[1, :] + cat_trim[2, :] + cat_trim[3, :] + + cat_trim[4, :], label = 'Erogazione finanziamento')
ax.bar(np.arange(4), cat_trim[6,:], bottom=cat_trim[0,:] + cat_trim[1, :] + cat_trim[2, :] + cat_trim[3, :] + + cat_trim[4, :] + cat_trim[5, :], label = 'Pagamento Fattura')

# Add labels, title, and legend
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['Trimestre 1', 'Trimestre 2', 'Trimestre 3', 'Trimestre 4'])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Distribuzione delle Categorie di Transazione su Base Trimestrale (%)')
ax.legend()

# Show the plot
plt.show()


# In[44]:


get_ipython().system('pip install pdfkit')
get_ipython().system('pip install jupyter_contrib_nbextensions')
get_ipython().system('jupyter nbextension enable --py --sys-prefix widgetsnbextension')
get_ipython().system('jupyter nbextension enable --py --sys-prefix qgrid')

import pdfkit
pdfkit.from_file('Assignment.ipynb', 'Assignment.pdf')


# In[ ]:




