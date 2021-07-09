#!/usr/bin/env python
# coding: utf-8

# # Comparative Analysis On Predicting Heart Diseases using Machine Learning                                         Algorithms   

# ### This Project is divided into 12 major steps which are as follows:
# 
# ##### 1. Importing Libraries & setting up environment
# ##### 2. Loading dataset
# ##### 3. Data Cleaning & Preprocessing
# ##### 4. Exploratory Data Analysis
# ##### 5. Outlier Detection & Removal
# ##### 6. Training & Test Split
# ##### 7. Cross Validation
# ##### 8. Model Building
# ##### 9. Model evaluation & comparison
# ##### 10. Feature Selection
# ##### 11. Model Evaluation
# ##### 12. Conclusion

# ## 1.Importing libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')

#data wrangling & pre-processing
import pandas as pd 
import numpy as np


# data visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


from sklearn.model_selection import train_test_split

#model validation
from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef
from sklearn import metrics

# cross validation
from sklearn.model_selection import StratifiedKFold

# machine learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import xgboost as xgb

    
from scipy import stats


# ## 2. Loading Dataset

# In[2]:


dt = pd.read_csv('final.csv')


# ### Lets see some of the sample entries of dataset

# In[3]:


dt.head()


# ##### "As we can see from above dataset entries some of the features should be nominal and to be encoded as their category type. In the next step we will be encoding features to their respective category as per the dataset description."

# ## 3.Data Cleaning and preprocessing

# ##### In this step we will first change the name of columns as some of the columns have weird naming pattern and then we will encode the features into categorical variables

# In[4]:


#renaming features to proper names
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope','target']


# In[5]:


dt.head()


# In[6]:


#converting features to categorical features
dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'
    
    
    
dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'
    
    
    
dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'
    
dt["sex"] = dt.sex.apply(lambda  x:'male' if x==1 else 'female')


# In[7]:


dt.head()


# In[8]:


dt['chest_pain_type'].value_counts()


# In[9]:


dt['rest_ecg'].value_counts()


# In[10]:


dt['st_slope'].value_counts()


# In[11]:


#dropping row with st_slope =0
dt.drop(dt[dt.st_slope ==0].index, inplace=True)
#checking distribution
dt['st_slope'].value_counts()


# In[12]:


# checking the top 5 entries of dataset after feature encoding
dt.head()


# ##### As we can see features are encoded sucessfully to their respective categories. Next we will be checking if there is any missing entry or not ?

# In[13]:


## Checking missing entries in the dataset columnwise
dt.isna().sum()


# ##### So, there are no missing entries in the dataset thats great. Next we will move towards exploring the dataset by performing detailed EDA

# ## 4.Exploratory Data Analysis (EDA)

# In[14]:


# first checking the shape of the dataset
dt.shape


# ##### So, there are total 1189 records and 11 features with 1 target variable. Lets check the summary of numerical and categorical features.

# In[15]:


# summary statistics of numerical columns
dt.describe(include =[np.number])


# ##### As we can see from above description resting_blood_pressure and cholestrol have some outliers as they have minimum value of 0 whereas cholestrol has outlier on upper side also having maximum value of 603.

# In[16]:


# summary statistics of categorical columns
dt.describe(include =[np.object])


# ### Distribution of Heart disease (target variable)

# In[17]:


# Plotting attrition of employees
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=False, figsize=(14,6))

ax1 = dt['target'].value_counts().plot.pie( x="Heart disease" ,y ='no.of patients', 
                    autopct = "%1.0f%%",labels=["Heart Disease","Normal"], startangle = 60,ax=ax1);
ax1.set(title = 'Percentage of Heart disease patients in Dataset')
    
ax2 = dt["target"].value_counts().plot(kind="barh" ,ax =ax2)
for i,j in enumerate(dt["target"].value_counts().values):
    ax2.text(.5,i,j,fontsize=12)
ax2.set(title = 'No. of Heart disease patients in Dataset')
plt.show()


# ##### The dataset is balanced having 628 heart disease patients and 561 normal patients.

# ### Checking Gender & Agewise Distribution

# In[18]:


plt.figure(figsize=(18,12))
plt.subplot(221)
dt["sex"].value_counts().plot.pie(autopct = "%1.0f%%",colors = sns.color_palette("prism",5),startangle = 60,labels=["Male","Female"],
wedgeprops={"linewidth":2,"edgecolor":"k"},explode=[.1,.1],shadow =True)
plt.title("Distribution of Gender")
plt.subplot(222)
ax= sns.distplot(dt['age'], rug=True)
plt.title("Age wise distribution")
plt.show()


# #### As we can see from above plot, in this dataset males percentage is way too higher than females where as average age of patients is around 55.

# In[19]:


# creating separate df for normal and heart patients

attr_1=dt[dt['target']==1]
    
attr_0=dt[dt['target']==0]
    
# plotting normal patients
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.distplot(attr_0['age'])
plt.title('AGE DISTRIBUTION OF NORMAL PATIENTS', fontsize=15, weight='bold')
    
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_0['sex'], palette='viridis')
plt.title('GENDER DISTRIBUTION OF NORMAL PATIENTS', fontsize=15, weight='bold' )
plt.show()

#plotting heart patients

fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.distplot(attr_1['age'])
plt.title('AGE DISTRIBUTION OF HEART DISEASE PATIENTS', fontsize=15, weight='bold')
    
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['sex'], palette='viridis')
plt.title('GENDER DISTRIBUTION OF HEART DISEASE PATIENTS', fontsize=15, weight='bold' )
plt.show()


# ##### As we can see from above plot more patients accounts for heart disease in comparison to females whereas mean age for heart disease patients is around 58 to 60 years.

# ### Distribution of Chest Pain Type

# In[20]:


# plotting normal patients
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['chest_pain_type'])
plt.title('CHEST PAIN OF NORMAL PATIENTS', fontsize=15, weight='bold')

#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['chest_pain_type'], palette='viridis')
plt.title('CHEST PAIN OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()


# In[21]:


#Exploring the Heart Disease patients based on Chest Pain Type
plot_criteria= ['chest_pain_type', 'target']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(dt[plot_criteria[0]], dt[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# #### As we can see from above plot  76% of the chest pain type of the heart disease patients have asymptomatic chest pain.  
# 

# ##### Asymptomatic heart attacks medically known as silent myocardial infarction (SMI) annually accounts for around 45-50% of morbidities due to cardiac ailments and even premature deaths in India. The incidences among middle aged people experiencing SMI is twice likely to develop in males than females. The symptoms of SMI being very mild in comparison to an actual heart attack; it is described as a silent killer. Unlike the symptoms in a normal heart attack which includes extreme chest pain, stabbing pain in the arms, neck & jaw, sudden shortness of breath, sweating and dizziness, the symptoms of SMI are very brief and hence confused with regular discomfort and most often ignored.

# ### Distribution of Rest ECG

# In[22]:


# plotting normal patients
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['rest_ecg'])
plt.title('REST ECG OF NORMAL PATIENTS', fontsize=15, weight='bold')
    
#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['rest_ecg'], palette='viridis')
plt.title('REST ECG OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()


# In[23]:


#Exploring the Heart Disease patients based on REST ECG
plot_criteria= ['rest_ecg', 'target']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(dt[plot_criteria[0]], dt[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# ##### An electrocardiogram records the electrical signals in your heart. It's a common test used to detect heart problems and monitor the heart's status in many situations. Electrocardiograms — also called ECGs or EKGs. but ECG has limits. It measures heart rate and rhythm—but it doesn’t necessarily show blockages in the arteries.Thats why in this dataset around 52% heart disease patients have normal ECG

# In[24]:


# plotting normal patients
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(attr_0['st_slope'])
plt.title('ST SLOPE OF NORMAL PATIENTS', fontsize=15, weight='bold')

#plotting heart patients
ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(attr_1['st_slope'], palette='viridis')
plt.title('ST SLOPE OF HEART PATIENTS', fontsize=15, weight='bold' )
plt.show()


# In[25]:


#Exploring the Heart Disease patients based on ST Slope
plot_criteria= ['st_slope', 'target']
cm = sns.light_palette("red", as_cmap=True)
(round(pd.crosstab(dt[plot_criteria[0]], dt[plot_criteria[1]], normalize='columns') * 100,2)).style.background_gradient(cmap = cm)


# ##### The ST segment /heart rate slope (ST/HR slope), has been proposed as a more accurate ECG criterion for diagnosing significant coronary artery disease (CAD) in most of the research papers. 

# ##### As we can see from above plot upsloping is positive sign as 74% of the normal patients have upslope where as 72.97% heart patients have flat sloping.

# ### Distribution of Numerical features

# In[26]:


sns.pairplot(dt, hue = 'target', vars = ['age', 'resting_blood_pressure', 'cholesterol'] )


# ##### From the above plot it is clear that as the age increases chances of heart disease increases

# In[27]:


sns.scatterplot(x = 'resting_blood_pressure', y = 'cholesterol', hue = 'target', data = dt)


# ##### From the above plot we can see outliers clearly as for some of the patients cholestrol is 0 whereas for one patient both cholestrol and resting bp is 0 which is may be due to missing entries we will filter these ouliers later

# In[28]:


sns.scatterplot(x = 'resting_blood_pressure', y = 'age', hue = 'target', data = dt)


# ## 5. Outlier Detection & Removal 

# ### Detecting outlier using z-score

# #####  An Outlier is am extreamly large or extreamly small data value relative to the rest of the data set.It may represent a data entry error,or it may be genuine data.

# # Z=x-u/sigma
# #### where
# ####           x=Score
# ####            u=Mean
# ####            sigma=Standard Deviation

# In[29]:


#filtering numeric features as age , resting bp, cholestrol and max heart rate achieved has outliers as per EDA

dt_numeric = dt[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved']]


# In[30]:


dt_numeric.head()


# In[31]:


# calculating zscore of numeric columns in the dataset
z = np.abs(stats.zscore(dt_numeric))
print(z)


# ##### from these points it is diffcult to say which points are outliers so we will now define threshold

# In[32]:


# Defining threshold for filtering outliers 
threshold = 3
print(np.where(z > 3))


# ##### Don’t be confused by the results. The first array contains the list of row numbers and second array respective column numbers, which mean  z[30][2]  have a Z-score higher than 3. There are total 17 data points which are outliers.

# In[33]:


#filtering outliers retaining only those data points which are below threshhold
dt = dt[(z < 3).all(axis=1)]


# In[34]:


# checking shape of dataset after outlier removal
dt.shape


# ##### All the 17 data points which are outliers are now removed.

# ##### Now before splitting dataset into train and test we first encode categorical variables as dummy variables and segregate feature and target variable.

# In[35]:


## encoding categorical variables
dt = pd.get_dummies(dt, drop_first=True)
dt.head()


# In[36]:


# checking the shape of dataset
dt.shape


# In[37]:


# segregating dataset into features i.e., X and target variables i.e., y
X = dt.drop(['target'],axis=1)
y = dt['target']


# ### Checking Correlation 

# In[38]:


#Correlation with Response Variable class

X.corrwith(y).plot.bar(
        figsize = (16, 4), title = "Correlation with Diabetes", fontsize = 15,
        rot = 90, grid = True)


# ## 6. Train Test Split 

# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)


# In[40]:


## checking distribution of traget variable in train test split\n",
print('Distribution of target variable in training set')
print(y_train.value_counts())

print('Distribution of target variable in test set')
print(y_test.value_counts())


# In[41]:


print('------------Training Set------------------')
print(X_train.shape)
print(y_train.shape)
    
print('------------Test Set------------------')
print(X_test.shape)
print(y_test.shape)


# ### feature normalization

# ##### In this step we will normalize all the numeric feature in the range of 0 to 1.

# In[42]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.fit_transform(X_train[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])
X_train.head()


# In[43]:


X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.transform(X_test[['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression']])
X_test.head()


# ## 7. Cross Validation 

# ##### In this step, we will build different baseline models and perform 10-fold cross validation to filter top performing baseline models to be used in level 0 of stacked ensemble method.

# In[44]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

#import xgboost as xgb
#function initializing baseline machine learning models
def GetBasedModel():
    basedModels = []
    basedModels.append(('NB'   , GaussianNB()))
    basedModels.append(('SVM Linear'  , SVC(kernel='linear',gamma='auto',probability=True)))
    basedModels.append(('Random forest'   , RandomForestClassifier(criterion='entropy',n_estimators=100)))

    return basedModels

# function for performing 10-fold cross validation of all the baseline models
def BasedLine2(X_train, y_train,models):
    # Test options and evaluation metric
    num_folds = 10
    scoring = 'accuracy'
    seed = 7
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=None)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    return results,msg


# In[45]:


models = GetBasedModel()
names,results = BasedLine2(X_train, y_train,models)


# ## 8. Model building

# ### Random Forest Classifier (criterion = 'entropy')

# In[46]:


rf_ent = RandomForestClassifier(criterion='entropy',n_estimators=100)
rf_ent.fit(X_train, y_train)
y_pred_rfe = rf_ent.predict(X_test)


# ### Support Vector Classifier (kernel='linear')

# In[47]:


svc = SVC(kernel='linear',gamma='auto',probability=True)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)


# ### Naive Bayes

# In[48]:


nb =GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)


# ## 9.Model Evaluation

# ##### In this step we will first define which evaluation metrics we will use to evaluate our model. The most important evaluation metric for this problem domain is  sensitivity, specificity, Precision, F1-measure, Geometric mean and mathew correlation coefficient and finally ROC AUC curve.

# ### Mathew Correlation coefficient (MCC)

# ##### The Matthews correlation coefficient (MCC), instead, is a more reliable statistical rate which produces a high score only if the prediction obtained good results in all of the four confusion matrix categories (true positives, false negatives, true negatives, and false positives), proportionally both to the size of positive elements and the size of negative elements in the dataset.
# #####        MCC=((TP+TN)-(FP+FN))/SQRT((TP+FP)+(TP+FN)-(TN+FP)-(TN+FN)
# #####        (worst value:-1;best value:+1)

# ### Log Loss

# ##### Logarithmic loss  measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0. Log loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high log loss.

# ### F1 Score

# ##### F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.
# ##### F1 Score = 2*(Recall * Precision) / (Recall + Precision)

# In[49]:


CM=confusion_matrix(y_test,y_pred_rfe)
sns.heatmap(CM, annot=True)
    
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
specificity = TN/(TN+FP)
loss_log = log_loss(y_test, y_pred_rfe)
acc= accuracy_score(y_test, y_pred_rfe)
roc=roc_auc_score(y_test, y_pred_rfe)
prec = precision_score(y_test, y_pred_rfe)
rec = recall_score(y_test, y_pred_rfe)
f1 = f1_score(y_test, y_pred_rfe)

mathew = matthews_corrcoef(y_test, y_pred_rfe)
model_results =pd.DataFrame([['Random Forest',acc, prec,rec,specificity, f1,roc, loss_log,mathew]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef'])
    
model_results


# ## Comparison with other Models

# In[50]:


data = {        'SVC': y_pred_svc,
                'GaussianNB': y_pred_nb}
                #'RandomForestClassifier': y_pred_rfe}
                
                
models = pd.DataFrame(data) 

for column in models:
    CM=confusion_matrix(y_test,models[column])
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    specificity = TN/(TN+FP)
    loss_log = log_loss(y_test, models[column])
    acc= accuracy_score(y_test, models[column])
    roc=roc_auc_score(y_test, models[column])
    prec = precision_score(y_test, models[column])
    rec = recall_score(y_test, models[column])
    f1 = f1_score(y_test, models[column])

    mathew = matthews_corrcoef(y_test, models[column])
    results =pd.DataFrame([[column,acc, prec,rec,specificity, f1,roc, loss_log,mathew]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef'])
    model_results = model_results.append(results, ignore_index = True)

model_results


# ### Findings

# #####  - As we can see from above results, Random Forest is best performer as it has highest test accuracy of 0.902128, sensitivity of 0.951220 and specificity of 0.848214 and highest f1-score of 0.910506 and lowest Log Loss of 3.3

# ## ROC AUC Curve

# In[51]:


def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

f, ax = plt.subplots(figsize=(12,8))
    
roc_auc_plot(y_test,rf_ent.predict_proba(X_test),label='Random Forest Classifier ',l='-')
roc_auc_plot(y_test,svc.predict_proba(X_test),label='Support Vector Machine ',l='-')
roc_auc_plot(y_test,nb.predict_proba(X_test),label='Naive Bayes',l='-')
    
ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
        )   
ax.legend(loc="lower right")    
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic curves')
sns.despine()


# ##### As we can see highest average area under the curve(AUC) of 0.946 is attained by Random Forest Classifier.

# ## Precision Recall curve

# In[52]:


def precision_recall_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_test,
                                                  y_proba[:,1])
    average_precision = average_precision_score(y_test, y_proba[:,1],
                                                     average="micro")
    ax.plot(recall, precision, label='%s (average=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)

f, ax = plt.subplots(figsize=(14,10))
    
precision_recall_plot(y_test,rf_ent.predict_proba(X_test),label='Random Forest Classifier ',l='-')
precision_recall_plot(y_test,svc.predict_proba(X_test),label='Support Vector Machine ',l='-')
precision_recall_plot(y_test,nb.predict_proba(X_test),label='Naive Bayes',l='-')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc="lower left")
ax.grid(True)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Precision-recall curves')
sns.despine()


# ## 10.  Feature Selection

# In[53]:


num_feats=11

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)
print(str(len(cor_feature)), 'selected features')


# In[54]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:,chi_support].columns.tolist()
print(str(len(chi_feature)), 'selected features')


# In[55]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:,rfe_support].columns.tolist()
print(str(len(rfe_feature)), 'selected features')


# In[56]:


from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
    
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2", solver='lbfgs'), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)
    
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
print(str(len(embeded_lr_feature)), 'selected features')


# In[57]:


from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
    
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, criterion='gini'), max_features=num_feats)
embeded_rf_selector.fit(X, y)
    
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)), 'selected features')


# In[58]:


from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
    
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    
embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
embeded_lgb_selector.fit(X, y)
    
embeded_lgb_support = embeded_lgb_selector.get_support()
embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
print(str(len(embeded_lgb_feature)), 'selected features')


# In[59]:


#put all selection together
feature_name = X.columns
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)


# In[60]:


# segregating dataset into features i.e., X and target variables i.e., y
X = dt.drop(['target','resting_blood_pressure','sex_male','chest_pain_type_non-anginal pain','chest_pain_type_atypical angina'],axis=1)
y = dt['target']


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2,shuffle=True, random_state=5)


# In[62]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[['age','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.fit_transform(X_train[['age','cholesterol','max_heart_rate_achieved','st_depression']])
X_train.head()


# In[63]:


X_test[['age','cholesterol','max_heart_rate_achieved','st_depression']] = scaler.transform(X_test[['age','cholesterol','max_heart_rate_achieved','st_depression']])
X_test.head()


# In[64]:


import xgboost as xgb
models = GetBasedModel()
names,results = BasedLine2(X_train, y_train,models)


# ## 11.Model Evaluation

# In[65]:


rf_ent = RandomForestClassifier(criterion='entropy',n_estimators=100)
rf_ent.fit(X_train, y_train)
y_pred_rfe = rf_ent.predict(X_test)


# In[66]:


svc = SVC(kernel='linear',gamma='auto',probability=True)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)


# In[67]:


nb =GaussianNB()
nb.fit(X_train,y_train)
y_pred_nb = nb.predict(X_test)


# In[68]:


data = {
             #'Random Forest Entropy': y_pred_rfe, 
             #   'SVC': y_pred_svc, 
             #   'GaussianNB': y_pred_nb
}
    
models = pd.DataFrame(data) 
    
for column in models:
    CM=confusion_matrix(y_test,models[column])
    
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    specificity = TN/(TN+FP)
    loss_log = log_loss(y_test, models[column])
    acc= accuracy_score(y_test, models[column])
    roc=roc_auc_score(y_test, models[column])
    prec = precision_score(y_test, models[column])
    rec = recall_score(y_test, models[column])
    f1 = f1_score(y_test, models[column])
    
    mathew = matthews_corrcoef(y_test, models[column])
    results =pd.DataFrame([[column,acc, prec,rec,specificity, f1,roc, loss_log,mathew]],
               columns = ['Model', 'Accuracy','Precision', 'Sensitivity','Specificity', 'F1 Score','ROC','Log_Loss','mathew_corrcoef'])
    model_results = model_results.append(results, ignore_index = True)
    
model_results


# ## Feature Importance

# In[69]:


feat_importances = pd.Series(rf_ent.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')


# ## 12.Conclusion 

# ##### From the above implementation we can come to a conclusion that Random Forest classifier has the high performance when compared with naive bayes and Support Vector Machine classification algorithms. 
# 
# 
# ##### 1. Out of the 11 features we examined, the top 5 significant features that helped us classify between a positive & negative Diagnosis were cholestrol, maximum heart rate achieved , age, ST depression and ST slope upsloping. 
# ##### 2. Our machine learning algorithm can now classify patients with Heart Disease. Now we can properly diagnose patients, & get them the help they needs to recover. By diagnosing detecting these features early, we may prevent worse symptoms from arising later.
# ##### 3. Our Random Forest algorithm yields the highest accuracy. Any accuracy above 70% is considered good, but be careful because if your accuracy is extremely high, it may be too good to be true (an example of Over fitting) and it cannot be too low. Thus, 85% is the ideal accuracy!
