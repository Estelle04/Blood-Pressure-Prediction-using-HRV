#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import heartpy as hp
import scipy.signal as ss
import scipy
import matplotlib
import math
import os
import glob
from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import LinAlgError


# # Data Processing

# In[2]:


path = r'C:\Files\NTU\URECA\Data\2023_header_data\2023_Data'
all_files = glob.glob(path + "/*csv")

li = []
file_name = []
for filename in all_files:
    filename_1 = os.path.basename(filename)
    filename_no_ext = os.path.splitext(filename_1)[0]
    file_name.append(filename_no_ext)
    #file_name.append(filename_1)
    df = pd.read_csv(filename, index_col = None, header = 0)
    li.append(df)


# In[3]:


ranking_df = pd.read_excel('Ranking.xlsx')


# In[4]:


def PPG_filtering(Dataset):
    timer = Dataset['Timestamp(ms)'].values
    sample_rate = len(Dataset)/((timer[-1]-timer[0])/1000)
    ppg_data = Dataset['PPG1']
    
    hf_ppg1 = hp.filter_signal(ppg_data, cutoff=[0.1,8], sample_rate=400, order=3, filtertype='bandpass')
    norm_ppg1 = scipy.special.softmax(ppg_data)
    norm_ppg = (norm_ppg1 - norm_ppg1.min()) / (norm_ppg1.max() - norm_ppg1.min())
    
    return norm_ppg

def PP_detection(data, time):
    firstder = np.gradient(data)
    secondder = np.gradient(firstder)
    peaks,_ = ss.find_peaks(data, height = 0.4, distance = 150)
    peaks_ts = time[peaks]
    
    rr_raw = np.diff(peaks_ts)
    pre_rr_corrected = rr_raw.copy()
    pre_rr_corrected[np.abs(zscore(rr_raw)) > 2] = np.median(rr_raw)
    q3, q1 = np.percentile(pre_rr_corrected, [75 ,25])
    rr_corrected = pre_rr_corrected[(pre_rr_corrected >= q1) & (pre_rr_corrected <= q3)]
    
    return rr_corrected


# In[5]:


ppg = []
PP = []

for filenumber in range(len(li)):
    filtered_PPG = PPG_filtering(li[filenumber])
    ppg.append(filtered_PPG)
    
    time = li[filenumber]['Timestamp(ms)'].values
    pp = PP_detection(ppg[filenumber], time)
    
    PP.append(pp)


# In[6]:


def mahalanobis(rr, cov = None):
    rr_n = rr[:-1]
    rr_n1 = rr[1:]
    md_df = pd.DataFrame({'rr1': rr_n, 'rr2': rr_n1})
    md_dff =  md_df[['rr1', 'rr2']]
    if not cov:
        cov = np.cov(md_dff.values.T)
    
    det = np.linalg.det(cov)
    eigvals, eigvecs = np.linalg.eig(cov)
    semi_axis_lengths = np.sqrt(eigvals)
    area = np.pi * np.sqrt(det) * semi_axis_lengths[0] * semi_axis_lengths[1]
    
    return area

def CHA(rr): 
    rr_n = rr[:-1]
    rr_n1 = rr[1:]
    coc = np.column_stack([rr_n, rr_n1])
    hull = ConvexHull(coc)
    area = hull.area
    
    return area

def calculate_rmssd(rr):
    differences = np.diff(rr)
    squared_differences = np.square(differences)
    mean_squared_diff = np.mean(squared_differences)
    rmssd = np.sqrt(mean_squared_diff)
    
    return rmssd

def CEP(rr): 
    rr_n = rr[:-1]
    rr_n1 = rr[1:]
    pre_distance_error_list = []
    rr_mean = np.mean(rr)
    pre_distance_error_list.append(rr - rr_mean)
    pre_distance_error = rr - rr_mean
    
    for dr_i in pre_distance_error_list:
        dr_Squared = dr_i**2
        dr_Squared_sum = sum(dr_Squared)
    
    dr_mean = sum(pre_distance_error)/len(pre_distance_error)
    dr_variance = dr_Squared_sum / len(pre_distance_error) - (dr_mean**2)
    dr_std_deviation = math.sqrt(dr_variance)
    cep_50 = 0.6745 * dr_std_deviation
    
    return cep_50


# In[7]:


MD_1 = []
RMSSD_1 = []
CHA_1 = []
CEP_1 = []

for filenumber in range(len(li)):
    pre_MD = mahalanobis(PP[filenumber])
    MD_1.append(pre_MD)
    
    pre_RMSSD = calculate_rmssd(PP[filenumber])
    RMSSD_1.append(pre_RMSSD)
    
    pre_CHA = CHA(PP[filenumber])
    CHA_1.append(pre_CHA)
    
    pre_CEP = CEP(PP[filenumber])
    CEP_1.append(pre_CEP)
    
scaler = MinMaxScaler()

MD_1_array = np.array(MD_1).reshape(-1, 1)

normalized_MD_1 = scaler.fit_transform(MD_1_array)

normalized_MD_1_list = normalized_MD_1.flatten().tolist()
    
rows = [(file_name[i], MD_1[i], RMSSD_1[i], CHA_1[i], CEP_1[i]) for i in range(len(li))]
result_df = pd.DataFrame(rows, columns=['Filename','MD area', 'RMSSD', 'CHA', 'CEP'])


# In[8]:


merged_df = result_df.merge(ranking_df, on='Filename')
merged_df = merged_df.drop(["Unnamed: 0", 'Consent Form No.', 'Last Meal (Hrs)', 'Systole_1st', 
                'Diastole_1st', 'Ht', 'Wt', 'Ethnic', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5', 'Final']
               , axis = 1)

merged_df


# In[10]:


male_df = merged_df[merged_df['Sex'] == 'M']
female_df = merged_df[merged_df['Sex'] == 'F']


# In[11]:


f, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].scatter(np.log(male_df['RMSSD']), np.log(male_df['MD area']), c='blue', label='Male')
ax[0].scatter(np.log(female_df['RMSSD']), np.log(female_df['MD area']), c='red', label='Female')
ax[0].set_xlabel('RMSSD(log scale)')
ax[0].set_ylabel('MD area(log scale)')
ax[0].legend()
ax[0].set_title('MD area against RMSSD (log scale)')

ax[1].scatter(male_df['RMSSD'], male_df['MD area'], c='blue', label='Male')
ax[1].scatter(female_df['RMSSD'], female_df['MD area'], c='red', label='Female')
ax[1].set_xlabel('RMSSD')
ax[1].set_ylabel('MD area(normalized)')
ax[1].legend()
ax[1].set_title('MD area against RMSSD')

log_x = np.log(merged_df['RMSSD'])
log_y = np.log(merged_df['MD area'])
correlation_coef = np.corrcoef(log_x, log_y)[0, 1]
correlation_coef_2 = np.corrcoef(merged_df['RMSSD'], merged_df['MD area'])[0, 1]
print("Log Correlation Coefficient:", correlation_coef)
print("Linear Correlation Coefficient2:", correlation_coef_2)


# In[12]:


f, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].scatter(np.log(male_df['RMSSD']), np.log(male_df['CHA']), c='blue', label='Male')
ax[0].scatter(np.log(female_df['RMSSD']), np.log(female_df['CHA']), c='red', label='Female')
ax[0].set_xlabel('RMSSD(log scale)')
ax[0].set_ylabel('CHA(log scale)')
ax[0].legend()
ax[0].set_title('CHA against RMSSD (log scale)')

ax[1].scatter(male_df['RMSSD'], male_df['CHA'], c='blue', label='Male')
ax[1].scatter(female_df['RMSSD'], female_df['CHA'], c='red', label='Female')
ax[1].set_xlabel('RMSSD')
ax[1].set_ylabel('CHA(normalized)')
ax[1].legend()
ax[1].set_title('CHA against RMSSD')

log_x = np.log(merged_df['RMSSD'])
log_y = np.log(merged_df['CHA'])
correlation_coef = np.corrcoef(log_x, log_y)[0, 1]
correlation_coef_2 = np.corrcoef(merged_df['RMSSD'], merged_df['CHA'])[0, 1]
print("Log Correlation Coefficient:", correlation_coef)
print("Linear Correlation Coefficient2:", correlation_coef_2)


# In[13]:


f, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].scatter(np.log(male_df['RMSSD']), np.log(male_df['CEP']), c='blue', label='Male')
ax[0].scatter(np.log(female_df['RMSSD']), np.log(female_df['CEP']), c='red', label='Female')
ax[0].set_xlabel('RMSSD(log scale)')
ax[0].set_ylabel('CEP(log scale)')
ax[0].legend()
ax[0].set_title('CEP against RMSSD (log scale)')

ax[1].scatter(male_df['RMSSD'], male_df['CEP'], c='blue', label='Male')
ax[1].scatter(female_df['RMSSD'], female_df['CEP'], c='red', label='Female')
ax[1].set_xlabel('RMSSD')
ax[1].set_ylabel('CEP(normalized)')
ax[1].legend()
ax[1].set_title('CEP against RMSSD')

log_x = np.log(merged_df['RMSSD'])
log_y = np.log(merged_df['CEP'])
correlation_coef = np.corrcoef(log_x, log_y)[0, 1]
correlation_coef_2 = np.corrcoef(merged_df['RMSSD'], merged_df['CEP'])[0, 1]
print("Log Correlation Coefficient:", correlation_coef)
print("Linear Correlation Coefficient2:", correlation_coef_2)


# # Trend

# ## Age

# In[14]:


fig, ax = plt.subplots(2, 2, figsize=(12,12))

clip_min = np.percentile(male_df['MD area'], 0)
clip_max = np.percentile(male_df['MD area'], 95)
male_df_clipped = male_df[(male_df['MD area'] >= clip_min) & (male_df['MD area'] <= clip_max)]
female_df_clipped = female_df[(female_df['MD area'] >= clip_min) & (female_df['MD area'] <= clip_max)]

ax[0, 0].scatter(male_df_clipped['Age'], male_df_clipped['MD area'], c='blue', label='Male')
ax[0, 0].scatter(female_df_clipped['Age'], female_df_clipped['MD area'], c='red', label='Female')
ax[0, 0].set_xlabel('Age')
ax[0, 0].set_ylabel('MD area')
ax[0, 0].legend()
ax[0, 0].set_title('MD area against Age')

ax[0, 1].scatter(male_df_clipped['Age'], male_df_clipped['CHA'], c='blue', label='Male')
ax[0, 1].scatter(female_df_clipped['Age'], female_df_clipped['CHA'], c='red', label='Female')
ax[0, 1].set_xlabel('Age')
ax[0, 1].set_ylabel('CHA')
ax[0, 1].legend()
ax[0, 1].set_title('CHA against Age')

ax[1, 0].scatter(male_df_clipped['Age'], male_df_clipped['CEP'], c='blue', label='Male')
ax[1, 0].scatter(female_df_clipped['Age'], female_df_clipped['CEP'], c='red', label='Female')
ax[1, 0].set_xlabel('Age')
ax[1, 0].set_ylabel('CEP')
ax[1, 0].legend()
ax[1, 0].set_title('CEP against Age')

ax[1, 1].scatter(male_df_clipped['Age'], male_df_clipped['RMSSD'], c='blue', label='Male')
ax[1, 1].scatter(female_df_clipped['Age'], female_df_clipped['RMSSD'], c='red', label='Female')
ax[1, 1].set_xlabel('Age')
ax[1, 1].set_ylabel('RMSSD')
ax[1, 1].legend()
ax[1, 1].set_title('RMSSD against Age')


# ## HRV

# In[15]:


fig, ax = plt.subplots(2, 2, figsize=(12,12))

clip_min = np.percentile(male_df['MD area'], 1)
clip_max = np.percentile(male_df['MD area'], 95)
male_df_clipped = male_df[(male_df['MD area'] >= clip_min) & (male_df['MD area'] <= clip_max)]
female_df_clipped = female_df[(female_df['MD area'] >= clip_min) & (female_df['MD area'] <= clip_max)]

ax[0, 0].scatter(male_df_clipped['HR_1st'], male_df_clipped['MD area'], c='blue', label='Male')
ax[0, 0].scatter(female_df_clipped['HR_1st'], female_df_clipped['MD area'], c='red', label='Female')
ax[0, 0].set_xlabel('HR_1st')
ax[0, 0].set_ylabel('MD area')
ax[0, 0].legend()
ax[0, 0].set_title('MD area against HR_1st')

ax[0, 1].scatter(male_df_clipped['HR_1st'], male_df_clipped['CHA'], c='blue', label='Male')
ax[0, 1].scatter(female_df_clipped['HR_1st'], female_df_clipped['CHA'], c='red', label='Female')
ax[0, 1].set_xlabel('HR_1st')
ax[0, 1].set_ylabel('CHA')
ax[0, 1].legend()
ax[0, 1].set_title('CHA against HR_1st')

ax[1, 0].scatter(male_df_clipped['HR_1st'], male_df_clipped['CEP'], c='blue', label='Male')
ax[1, 0].scatter(female_df_clipped['HR_1st'], female_df_clipped['CEP'], c='red', label='Female')
ax[1, 0].set_xlabel('HR_1st')
ax[1, 0].set_ylabel('CEP')
ax[1, 0].legend()
ax[1, 0].set_title('CEP against HR_1st')

ax[1, 1].scatter(male_df_clipped['HR_1st'], male_df_clipped['RMSSD'], c='blue', label='Male')
ax[1, 1].scatter(female_df_clipped['HR_1st'], female_df_clipped['RMSSD'], c='red', label='Female')
ax[1, 1].set_xlabel('HR_1st')
ax[1, 1].set_ylabel('RMSSD')
ax[1, 1].legend()
ax[1, 1].set_title('RMSSD against HR_1st')


# ## BGL

# In[18]:


fig, ax = plt.subplots(2, 2, figsize=(12,12))

clip_min = np.percentile(male_df['MD area'], 0)
clip_max = np.percentile(male_df['MD area'], 95)
male_df_clipped = male_df[(male_df['MD area'] >= clip_min) & (male_df['MD area'] <= clip_max)]
female_df_clipped = female_df[(female_df['MD area'] >= clip_min) & (female_df['MD area'] <= clip_max)]

ax[0, 0].scatter(male_df_clipped['BGL (mg/dL)'], male_df_clipped['MD area'], c='blue', label='Male')
ax[0, 0].scatter(female_df_clipped['BGL (mg/dL)'], female_df_clipped['MD area'], c='red', label='Female')
ax[0, 0].set_xlabel('BGL (mg/dL)')
ax[0, 0].set_ylabel('MD area')
ax[0, 0].legend()
ax[0, 0].set_title('MD area against BGL')

ax[0, 1].scatter(male_df_clipped['BGL (mg/dL)'], male_df_clipped['CHA'], c='blue', label='Male')
ax[0, 1].scatter(female_df_clipped['BGL (mg/dL)'], female_df_clipped['CHA'], c='red', label='Female')
ax[0, 1].set_xlabel('BGL (mg/dL)')
ax[0, 1].set_ylabel('CHA')
ax[0, 1].legend()
ax[0, 1].set_title('CHA against BGL')

ax[1, 0].scatter(male_df_clipped['BGL (mg/dL)'], male_df_clipped['CEP'], c='blue', label='Male')
ax[1, 0].scatter(female_df_clipped['BGL (mg/dL)'], female_df_clipped['CEP'], c='red', label='Female')
ax[1, 0].set_xlabel('BGL (mg/dL)')
ax[1, 0].set_ylabel('CEP')
ax[1, 0].legend()
ax[1, 0].set_title('CEP against BGL')

ax[1, 1].scatter(male_df_clipped['BGL (mg/dL)'], male_df_clipped['RMSSD'], c='blue', label='Male')
ax[1, 1].scatter(female_df_clipped['BGL (mg/dL)'], female_df_clipped['RMSSD'], c='red', label='Female')
ax[1, 1].set_xlabel('BGL (mg/dL)')
ax[1, 1].set_ylabel('RMSSD')
ax[1, 1].legend()
ax[1, 1].set_title('RMSSD against BGL')


# ## BMI

# In[17]:


fig, ax = plt.subplots(2, 2, figsize=(12,12))

clip_min = np.percentile(male_df['MD area'], 0)
clip_max = np.percentile(male_df['MD area'], 95)
male_df_clipped = male_df[(male_df['MD area'] >= clip_min) & (male_df['MD area'] <= clip_max)]
female_df_clipped = female_df[(female_df['MD area'] >= clip_min) & (female_df['MD area'] <= clip_max)]

ax[0, 0].scatter(male_df_clipped['BMI'], male_df_clipped['MD area'], c='blue', label='Male')
ax[0, 0].scatter(female_df_clipped['BMI'], female_df_clipped['MD area'], c='red', label='Female')
ax[0, 0].set_xlabel('BMI')
ax[0, 0].set_ylabel('MD area')
ax[0, 0].legend()
ax[0, 0].set_title('MD area against BMI')

ax[0, 1].scatter(male_df_clipped['BMI'], male_df_clipped['CHA'], c='blue', label='Male')
ax[0, 1].scatter(female_df_clipped['BMI'], female_df_clipped['CHA'], c='red', label='Female')
ax[0, 1].set_xlabel('BMI')
ax[0, 1].set_ylabel('CHA')
ax[0, 1].legend()
ax[0, 1].set_title('CHA against BMI')

ax[1, 0].scatter(male_df_clipped['BMI'], male_df_clipped['CEP'], c='blue', label='Male')
ax[1, 0].scatter(female_df_clipped['BMI'], female_df_clipped['CEP'], c='red', label='Female')
ax[1, 0].set_xlabel('BMI')
ax[1, 0].set_ylabel('CEP')
ax[1, 0].legend()
ax[1, 0].set_title('CEP against BMI')

ax[1, 1].scatter(male_df_clipped['BMI'], male_df_clipped['RMSSD'], c='blue', label='Male')
ax[1, 1].scatter(female_df_clipped['BMI'], female_df_clipped['RMSSD'], c='red', label='Female')
ax[1, 1].set_xlabel('BMI')
ax[1, 1].set_ylabel('RMSSD')
ax[1, 1].legend()
ax[1, 1].set_title('RMSSD against BMI')


# # Machine learning

# In[19]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# In[20]:


condition_list = []

for filenumber in range(len(li)):
    if 11 < RMSSD_1[filenumber] < 87:
        condition_list.append('Healthy')
    elif (110< RMSSD_1[filenumber]):
        condition_list.append('Error')
    else:
        condition_list.append('Concerned')

result_df['Condition'] = condition_list


# In[23]:


x = np.array(result_df[['CHA', 'CEP','MD area']]) 
y_classification = result_df['Condition']

x_train, x_test, y_train_classification, y_test_classification = train_test_split(x, y_classification, test_size=0.25, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(x_train, y_train_classification)
y_pred_classification = rf_classifier.predict(x_test)
accuracy = accuracy_score(y_test_classification, y_pred_classification)
print("Classification Accuracy:", accuracy)


# In[25]:


X = np.array(result_df[['CHA', 'CEP','MD area']]) 
y = result_df['RMSSD'].values

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

df3 = pd.DataFrame([], columns= [])
df3['predicted rMSSD'] = rf.predict(X_test)
df3['rMSSD'] = y_test
df3['error'] = abs(df3['rMSSD']-df3['predicted rMSSD'])
df3['predicted condition'] = rf_classifier.predict(X_test)
df3['predicted rMSSD'] = df3['predicted rMSSD'].apply(lambda x: float("{:.1f}".format(x)))
df3['error'] = df3['error'].apply(lambda x: float("{:.1f}".format(x)))
df3


# In[ ]:




