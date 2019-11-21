#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
with zipfile.ZipFile('histopathologic-cancer-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('data')


# In[3]:


import numpy as np
import pandas as pd
from glob import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from sklearn.utils import shuffle
from tqdm import tqdm_notebook


# In[4]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


from fastai.vision import *
from fastai.metrics import error_rate


# In[6]:


path=Path('~/practicum/kaggle/data/').expanduser()
train_path=path/'train/'
test_path=path/'test/'
train_label=path/'train_labels.csv'
ORG_SIZE=96

bs=64
num_workers=None # Apprently 2 cpus per kaggle node, so 4 threads I think
sz=96


# In[7]:


train_lbl = pd.read_csv(train_label)


# In[8]:


train_lbl['label'].value_counts()


# # Visualization_2 (fast.ai)

# In[9]:


fnames = get_image_files(train_path)
fnames[:5]


# In[10]:


from sklearn.model_selection import train_test_split

# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform
train_df = train_lbl.set_index('id')

train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

# split, this function returns more than we need as we only need the validation indexes for fastai
tr_n, val_n, tr_idx,val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)


# In[11]:


arch = models.resnet34                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment                     # input size is the crop size
MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50'


# In[12]:


train_dict = {'name': train_names, 'label': train_labels}
df = pd.DataFrame(data=train_dict)
# create test dataframe
test_names = []
for f in os.listdir(test_path):
    test_names.append(f)
df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])


# In[13]:


df['name']


# In[14]:


data = ImageList.from_df(path=train_path ,
                              df=df, suffix = '.tif')\
.split_by_idx(val_idx)\
.label_from_df(cols='label')\
.add_test(ImageList.from_df(path=test_path, df=df_test))\
.transform(tfms=[[],[]], size=sz)\
.databunch(bs=bs)

    
    


# In[15]:


data.show_batch(rows=2, figsize=(7,6))


# # Training: resnet34

# In[16]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[17]:


learn.model


# In[18]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1')


# # Results

# In[63]:


learn.unfreeze()


# In[64]:


learn.fit_one_cycle(1)


# In[19]:


learn.load('stage-1');


# In[66]:


### learning rate finder
learn.lr_find()


# In[67]:


learn.recorder.plot()


# In[68]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[20]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(title='Confusion matrix')


# In[21]:


learn.save('stage-1');


# In[22]:


preds,y, loss = learn.get_preds(with_loss=True)
# get accuracy
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))


# # submit

# In[42]:


sample_df = pd.read_csv(path/'sample_submission.csv')
sample_list = list(sample_df.id)

# List of tumor preds. 
# These are in the order of our test dataset and not necessarily in the same order as in sample_submission
preds,y = learn.get_preds(ds_type=DatasetType.Test, with_loss=False)
tumor_preds = preds[:, 1]
pred_list = [p for p in tumor_preds]

# To know the id's, we create a dict of id:pred
pred_dic = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items, pred_list))

# Now, we can create a new list with the same order as in sample_submission
pred_list_cor = [pred_dic['/home/linqisheng/practicum/kaggle/data/test/'+ id +'.tif'] for id in sample_list]

# Next, a Pandas dataframe with id and label columns.
df_sub = pd.DataFrame({'id':sample_list,'label':pred_list_cor})

# Export to csv
df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)


# In[43]:


df_sub.head()


# In[46]:


from IPython.display import FileLink
FileLink(r'resnet34_submission.csv')


# In[ ]:




