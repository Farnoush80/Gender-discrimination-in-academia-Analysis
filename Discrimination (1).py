#!/usr/bin/env python
# coding: utf-8

# It is believed that discrimination in academia has been decreased over time. However, gender discrimination is truly a subtle form of prejudice that has a striking impact on all aspects of life. The level of gender discrimination has been analyzed in the five different departments of the Houston College of Medicine. Female doctors claimed that the College has engaged in a pattern and practice of discrimination against women in giving promotions and setting salaries. 
# Data explorations helps to reveal hidden patterns in raw data and well understand the relation between features. In terms of different departments involved in this claim, analyzing each department is indispensable to perceive the level and type of gender gap in medical school. 
# 

# In[1]:


import numpy as np
import pandas as pd

import seaborn as sns;sns.set()
from sklearn. model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# In[2]:


Dis=pd.read_excel('C:\\Users\\noosh\\Documents\\Custom Office Templates\\Lawsuit.xlt')


# In[3]:


Dis.drop('ID',1,inplace=True)


# General outlook of data

# In[4]:


pd.set_option('display.max_rows',500)


# In[5]:


Dis.head()


# In[6]:


Dis.tail()


# In[7]:


Dis.info()


# Data Cleaning

# In[8]:



Dum = pd.get_dummies(Dis. Clin, prefix='Clin').iloc[:]
Dis1= pd.concat([Dis,Dum],axis=1)
Dum1= pd.get_dummies(Dis1.Cert , prefix='Cert').iloc[:]
Dis2= pd.concat([Dis1,Dum1],axis=1)
Dum2= pd.get_dummies(Dis2.Rank, prefix='Rank').iloc[:]
DIS = pd.concat([Dis2,Dum2],axis=1)


# In[9]:


DIS.head()


# In[10]:


DIS.rename(columns={'Clin_0':'Primarily_research_emphasis','Clin_1':'Primarily_clinical_emphasis','Cert_0':'not_certified','Cert_1':'Board_certified','Rank_1':'Assistant','Rank_2':'Associate','Rank_3':'full_professor'},inplace=True)


# In[11]:


DIS.drop(['Clin','Cert','Rank'],1,inplace=True)


# In[12]:


DIS.head()


# In[13]:


DIS.dtypes


# In[14]:


DIS=DIS.replace(',.','',regex=True).astype('float')


# In[15]:


DIS.isnull().sum().reset_index()


# In[16]:


DIS[DIS['Exper']<30]


# 1=Dept 
# __Biochemistry/Molecular Biology__
# ,__Physiology__
# ,__Genetics__
# ,__Pediatrics__
# ,__Medicine__
# ,__Surgery__
# 
# 2 Gender __1=Male, 0=Female__
# 
# 3 Clin __1=Primarily clinical emphasis, 0=Primarily research emphasis__
# 
# 4 Cert __1=Board certified, 0=not certified__
# 
# 5 __Prate Publication rate__ (# publications on cv)/(# years between CV date and MD date)
# 
# 6 __Exper__ # years since obtaining MD
# 
# 7 Rank __1=Assistant, 2=Associate, 3=Full professor__ (a proxy for productivity)
# 
# 8 Sal94 Salary in academic year 1994
# 
# 9 Sal95 Salary after increment to 1994

# __Biochemistry__

# In[17]:


Biochemistry=DIS.loc[1:49,]


# In[18]:


Biochemistry[Biochemistry['Exper']<30].head()


# In[19]:


Biochemistry.Gender.value_counts(normalize=True)


# In[20]:


Biochemistry.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=14,color='darkblue')
plt.legend(['0=Female'])


# In[21]:


Biochemistry.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=12,labelpad=6,color='darkblue')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# In[22]:


Biochemistry.groupby('Gender').Associate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=12,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# In[23]:


Biochemistry.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=12,labelpad=6,color='darkblue')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# Funders of medical research the world over are increasingly seeking, in research assessment, to complement traditional output measures of scientific publications with more outcome-based indicators of societal and economic impact

# In[24]:


Biochemistry.groupby('Gender').Primarily_research_emphasis.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_research_emphasis',size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# In[25]:


Biochemistry.groupby('Gender').Primarily_clinical_emphasis.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_clinical_emphasis',size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# In[26]:


Biochemistry.Assistant.value_counts(normalize=True)


# Funders of medical research the world over are increasingly seeking, in research assessment, to complement traditional output measures of scientific publications with more outcome-based indicators of societal and economic impact

# In[27]:


Biochemistry.groupby('Primarily_research_emphasis').Gender.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Assistant',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='darkblue')
plt.show()


# In[28]:


Biochemistry.groupby('Primarily_clinical_emphasis').Assistant.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Primarily_clinical_emphasis',size=14,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.show()


# In[29]:


Biochemistry.groupby('Primarily_clinical_emphasis').Associate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Primarily_clinical_emphasis',size=14,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.show()


# In[30]:


Biochemistry.groupby('Primarily_research_emphasis').Associate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Associate',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='darkblue')
plt.show()


# In[31]:


Biochemistry.groupby('Exper').Assistant.mean().plot(kind='bar',width=0.4,color='lightblue',edgecolor='deepskyblue',figsize=(8,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkblue')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkblue')
plt.show()


# In[ ]:


Biochemistry.groupby('Exper').Associate.mean().plot(kind='bar',width=0.4,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.show()


# In[33]:


Biochemistry.groupby(['Gender','Primarily_research_emphasis']).Assistant.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Assistant',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='darkblue')
plt.show()


# In[34]:


Biochemistry.groupby(['Gender','Primarily_research_emphasis']).Associate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Associate',size=14,labelpad=6,color='darkblue')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='darkblue')
plt.show()


# In[35]:


Biochemistry.groupby('Exper').full_professor.mean().plot(kind='bar',width=0.3,color='skyblue',edgecolor='deepskyblue',figsize=(10,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.show()


# In[36]:


Biochemistry.groupby(['Gender','Exper']).full_professor.median().plot(kind='bar',width=0.6,color='skyblue',edgecolor='deepskyblue',figsize=(10,4))


# In[37]:


Biochemistry.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Board_certified', size=14,labelpad=6,color='darkblue')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkblue')
plt.show()


# In[38]:


Biochemistry.groupby('Board_certified').Associate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkblue')
plt.ylabel('Associate', size=14,labelpad=6,color='darkblue')
plt.legend(['1=Board_certified'])
plt.show()


# In[39]:


Biochemistry.groupby('Board_certified').full_professor.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkblue')
plt.ylabel('full_professor', size=14,labelpad=6,color='darkblue')
plt.legend(['1=Board_certified'])
plt.show()


# In[40]:


Biochemistry.groupby('Gender').Prate.mean().plot(kind='bar',width=0.2,color='lightblue',edgecolor='deepskyblue',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkblue')
plt.ylabel(' Prate publication rate', size=14,labelpad=6,color='darkblue')
plt.legend(['0=Female'])
plt.show()


# __Physiology__

# In[41]:


Physiology=DIS.loc[50:89,]


# In[42]:


Physiology[Physiology['Exper']<30].head()


# In[43]:


Physiology.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,color='tomato')
plt.legend(['0=Female'])


# In[44]:


Physiology.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('Assistant', size=14,labelpad=6,color='tomato')
plt.show()


# In[45]:


Physiology.groupby('Gender').Associate.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('Associate', size=14,labelpad=6,color='tomato')
plt.legend(['0=Fmale'])
plt.show()


# In[46]:


Physiology.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('full proffesor', size=14,labelpad=6,color='tomato')
plt.show()


# In[47]:


Physiology.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='tomato')
plt.ylabel('Associate', size=14,labelpad=6,color='tomato')
plt.legend(['1=Board_certified'])
plt.show()


# In[48]:


Physiology.groupby('Board_certified').Associate.mean().plot(kind='bar',width=0.2,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='tomato')
plt.ylabel('Associate', size=14,labelpad=6,color='tomato')
plt.legend(['1=Board_certified'])
plt.show()


# In[49]:


Physiology.groupby('Board_certified').full_professor.mean().plot(kind='bar',width=0.2,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='tomato')
plt.ylabel('full_professor', size=14,labelpad=6,color='tomato')
plt.legend(['1=Board_certified'])
plt.show()


# In[50]:


Physiology.groupby('Gender').Primarily_research_emphasis.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('Primiarily_research_emphasis', size=14,labelpad=6,color='tomato')
plt.show()


# In[51]:


Physiology.groupby('Gender').Primarily_clinical_emphasis.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('Primarily_clinical_emphasis', size=14,labelpad=6,color='tomato')
plt.show()


# In[52]:


Physiology.groupby('Gender').Prate.mean().plot(kind='bar',width=0.1,color='coral',edgecolor='orangered',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('Publication_rate', size=14,labelpad=6,color='tomato')
plt.legend(['1=Male'])
plt.show()


# In[53]:


Physiology.groupby('Exper').Assistant.mean().plot(kind='bar',width=0.3,color='coral',edgecolor='orangered',figsize=(8,4))
plt.xlabel('Years sience obtaining MD',size=14,labelpad=6,color='tomato')
plt.ylabel('Assistant', size=14,labelpad=6,color='tomato')
plt.show()


# In[54]:


Physiology.groupby('Exper').Associate.mean().plot(kind='bar',width=0.3,color='coral',edgecolor='orangered',figsize=(8,4))
plt.xlabel('Years sience obtaining MD',size=13,labelpad=6,color='tomato')
plt.ylabel('Associate', size=14,labelpad=6,color='tomato')
plt.show()


# In[55]:


Physiology.groupby('Exper').full_professor.mean().plot(kind='bar',width=0.2,color='coral',edgecolor='orangered',figsize=(8,4))
plt.xlabel('Gender',size=14,labelpad=6,color='tomato')
plt.ylabel('full_professor', size=14,labelpad=6,color='tomato')
plt.show()


# In[56]:


Physiology.groupby(['Gender','Exper']).full_professor.median().plot(kind='bar',width=0.3,color='coral',edgecolor='orangered',figsize=(10,4))


# __Genetics__

# In[57]:


Genetics=DIS.loc[90:110,]


# In[58]:


Genetics.head()


# In[59]:


Genetics.Gender.value_counts(normalize=True)


# In[60]:


Genetics.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=12,color='darkolivegreen')
plt.legend(['0=Female'])


# In[61]:


Genetics.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['0=Female'])
plt.show()


# In[62]:


Genetics.groupby('Gender').Associate.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Associate', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['0=Fmale'])
plt.show()


# In[63]:


Genetics.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['0=Fmale'])
plt.show()


# In[64]:


Genetics.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# In[65]:


Genetics.groupby('Board_certified').Associate.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Associate', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['1=Board_certified'])
plt.show()


# In[66]:


Genetics.groupby('Board_certified').full_professor.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('full_professor', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['1=Board_certified'])
plt.show()


# In[67]:


Genetics.groupby('Gender').Primarily_research_emphasis.mean().plot(kind='bar',width=0.1,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Primiarily_research_emphasis', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# In[68]:


Genetics.groupby('Gender').Primarily_clinical_emphasis.mean().plot(kind='bar',width=0.1,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Primarily_clinical_emphasis', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# In[69]:


Genetics.groupby('Gender').Prate.mean().plot(kind='bar',width=0.1,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Publication_rate', size=14,labelpad=6,color='darkolivegreen')
plt.legend(['0=Female'])
plt.show()


# In[70]:


Genetics.groupby('Exper').Assistant.mean().plot(kind='bar',width=0.3,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# In[71]:


Genetics.groupby('Exper').Associate.mean().plot(kind='bar',width=0.2,color='lightyellow',edgecolor='gold',figsize=(6,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('Associate', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# In[72]:


Biochemistry.groupby('Exper').full_professor.mean().plot(kind='bar',width=0.3,color='lightyellow',edgecolor='gold',figsize=(8,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkolivegreen')
plt.ylabel('full_professor', size=14,labelpad=6,color='darkolivegreen')
plt.show()


# __Pediatrics__

# In[73]:


Pediatrics=DIS.loc[111:140,]


# In[74]:


Pediatrics.head()


# In[75]:


Pediatrics.Gender.value_counts(normalize=True)


# In[76]:


Pediatrics.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Gender',size=14,color='indigo')
plt.legend(['0=Female'])


# In[77]:


Pediatrics.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.1,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='indigo')
plt.ylabel('Assistant', size=14,labelpad=6,color='indigo')
plt.legend(['0=Female'])
plt.show()


# In[78]:


Pediatrics.groupby('Gender').Associate.mean().plot(kind='bar',width=0.1,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='indigo')
plt.ylabel('Associate', size=14,labelpad=6,color='indigo')
plt.legend(['0=Female'])
plt.show()


# In[79]:


Pediatrics.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.1,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='indigo')
plt.ylabel('full_professor', size=14,labelpad=6,color='indigo')
plt.legend(['0=Female'])
plt.show()


# In[80]:


Pediatrics.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='indigo')
plt.ylabel('Assistant', size=14,labelpad=6,color='indigo')
plt.show()


# In[81]:


Pediatrics.groupby('Board_certified').Associate.mean().plot(kind='bar',width=0.2,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='indigo')
plt.ylabel('Associate', size=14,labelpad=6,color='indigo')
plt.legend(['1=Board_certified'])
plt.show()


# In[82]:


Pediatrics.groupby('Board_certified').full_professor.mean().plot(kind='bar',width=0.2,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='indigo')
plt.ylabel('full_professor', size=14,labelpad=6,color='indigo')
plt.legend(['1=Board_certified'])
plt.show()


# In[83]:


Pediatrics.groupby('Primarily_research_emphasis').Gender.mean().plot(kind='bar',width=0.2,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Assistant',size=14,labelpad=6,color='indigo')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='indigo')
plt.show()


# In[84]:


Pediatrics.groupby('Primarily_clinical_emphasis').Gender.mean().plot(kind='bar',width=0.2,color='violet',edgecolor='purple',figsize=(6,4))
plt.xlabel('Assistant',size=14,labelpad=6,color='indigo')
plt.ylabel('Primarily_research_emphasis', size=14,labelpad=6,color='indigo')
plt.show()


# In[85]:


Pediatrics.groupby('Exper').Assistant.mean().plot(kind='bar',width=0.3,color='violet',edgecolor='purple',figsize=(8,4))
plt.xlabel('years since obtaining MD',size=14,labelpad=6,color='indigo')
plt.ylabel('Associate', size=14,labelpad=6,color='indigo')
plt.show()


# In[86]:


Pediatrics.groupby('Exper').Associate.mean().plot(kind='bar',width=0.3,color='violet',edgecolor='purple',figsize=(8,4))
plt.xlabel('years since obtaining MD',size=14,labelpad=6,color='indigo')
plt.ylabel('Associate', size=14,labelpad=6,color='indigo')
plt.show()


# In[87]:


Pediatrics.groupby(['Gender','Exper']).full_professor.mean().plot(kind='bar',width=0.3,color='violet',edgecolor='purple',figsize=(8,4))
plt.xlabel('years since obtaining MD',size=14,labelpad=6,color='indigo')
plt.ylabel('full_professor', size=14,labelpad=6,color='indigo')
plt.show()


# __Medicine__

# In[88]:


Medicine=DIS.loc[141:220,]


# In[89]:


Medicine[Medicine['Exper']<30].head()


# In[90]:


Medicine.Gender.value_counts(normalize=True)


# In[91]:


Medicine.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='red',figsize=(4,6))
plt.xlabel('Gender',size=14,color='darkred')
plt.legend(['0=Female'])


# In[92]:


Medicine.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.1,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkred')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkred')
plt.legend(['0=Female'])
plt.show()


# In[93]:


Medicine.groupby('Gender').Associate.mean().plot(kind='bar',width=0.1,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkred')
plt.ylabel('Associate', size=14,labelpad=6,color='darkred')
plt.legend(['0=Female'])
plt.show()


# In[94]:


Medicine.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.2,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkred')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkred')
plt.legend(['0=Fmale'])
plt.show()


# In[95]:


Medicine.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='red',edgecolor='gold',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkred')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkred')
plt.legend(['Board_certifited=1'])
plt.show()


# In[96]:


Medicine.groupby('Board_certified').full_professor.mean().plot(kind='bar',width=0.2,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='darkred')
plt.ylabel('full_professor', size=14,labelpad=6,color='darkred')
plt.legend(['1=Board_certified'])
plt.show()


# In[97]:


Medicine.groupby('Gender').Primarily_research_emphasis.mean().plot(kind='bar',width=0.1,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkred')
plt.ylabel('Primiarily_research_emphasis', size=14,labelpad=6,color='darkred')
plt.show()


# In[98]:


Medicine.groupby('Gender').Primarily_clinical_emphasis.mean().plot(kind='bar',width=0.1,color='red',edgecolor='darkred',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='darkred')
plt.ylabel('Primarily_clinical_emphasis', size=14,labelpad=6,color='darkred')
plt.show()


# In[99]:


Medicine.groupby('Exper').Assistant.mean().plot(kind='bar',width=0.3,color='red',edgecolor='darkred',figsize=(8,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkred')
plt.ylabel('Assistant', size=14,labelpad=6,color='darkred')
plt.show()


# In[100]:


Medicine.groupby('Exper').Associate.mean().plot(kind='bar',width=0.2,color='red',edgecolor='darkred',figsize=(8,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkred')
plt.ylabel('Associate', size=14,labelpad=6,color='darkred')
plt.show()


# In[101]:


Medicine.groupby('Exper').full_professor.mean().plot(kind='bar',width=0.3,color='red',edgecolor='darkred',figsize=(8,4))
plt.xlabel('Years since obtaining MD',size=14,labelpad=6,color='darkred')
plt.ylabel('full_professor', size=14,labelpad=6,color='darkred')
plt.show()


# __Surgery__

# In[102]:


Surgery=DIS.loc[221:260,]


# In[103]:


Surgery.head()


# In[104]:


Surgery.Gender.value_counts(normalize=True)


# In[105]:


Surgery.Gender.value_counts(normalize=True).plot(kind='bar',width=0.2, color='silver',edgecolor='dimgray',figsize=(4,6))
plt.xlabel('Gender',size=14,color='black')
plt.legend(['0=Female'])


# In[106]:


Surgery.groupby('Gender').Assistant.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Assistant', size=14,labelpad=6,color='black')
plt.legend(['0=Female'])
plt.show()


# In[107]:


Surgery.groupby('Gender').Associate.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Associate', size=14,labelpad=6,color='black')
plt.legend(['1=Male'])
plt.show()


# In[108]:


Surgery.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Assistant', size=14,labelpad=6,color='black')
plt.legend(['0=Female'])
plt.show()


# In[109]:


Surgery.groupby('Board_certified').Associate.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='black')
plt.ylabel('Associate', size=14,labelpad=6,color='black')
plt.legend(['1=Board_certified'])
plt.show()


# In[110]:


Surgery.groupby('Gender').Prate.mean().plot(kind='bar',width=0.1,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Publication_rate', size=14,labelpad=6,color='black')
plt.legend(['0=Female'])
plt.show()


# In[111]:


Surgery.groupby('Gender').Primarily_research_emphasis.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=12,labelpad=6,color='black')
plt.ylabel('Primarily_research_emphasis',size=12,labelpad=6,color='black')
plt.legend(['1=Male','0=female'])
plt.show()


# In[112]:


Surgery.groupby('Gender').Primarily_clinical_emphasis.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Primarily_clinical_emphasis',size=14,labelpad=6,color='black')
plt.legend(['1=Male','0=female'])
plt.show()


# In[113]:


Surgery.groupby('Board_certified').Assistant.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Board_certified',size=14,labelpad=6,color='black')
plt.ylabel('Assistant', size=14,labelpad=6,color='black')
plt.show()


# In[114]:


Surgery.groupby('Gender').full_professor.mean().plot(kind='bar',width=0.2,color='silver',edgecolor='dimgray',figsize=(6,4))
plt.xlabel('Gender',size=14,labelpad=6,color='black')
plt.ylabel('Assistant', size=14,labelpad=6,color='black')
plt.legend(['1=Male'])
plt.show()

