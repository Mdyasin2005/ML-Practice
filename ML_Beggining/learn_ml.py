import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import pearsonr as pearesonr
warnings.filterwarnings("ignore")
df=pd.read_csv('insurance.csv')
df.info()
#print(df.describe())
#print(df.isnull().sum())
numeric_coloumn= ['age','bmi','children','charges']
df_clean=df.copy()
#print(df_clean.shape)
df_clean.drop_duplicates(inplace=True)
#print(df_clean.shape)
#print(df_clean['sex'].value_counts())
df_clean['sex']=df_clean['sex'].map({'male':0,'female':1})
df_clean['smoker']=df_clean['smoker'].map({'yes':1,'no':0})
#print(df_clean.head())
#print(df['region'].value_counts())
df_clean=pd.get_dummies(df_clean,columns= ['region'],drop_first=True)
#print(df_clean.head())
df_clean=df_clean.astype(int)
#print(df_clean.head())
df_clean['bmi_catagory']=pd.cut(
    df_clean['bmi'],
    bins=[0,18.5,24.9,29.9,float('inf')],
    labels=['underweight','normal','overweight','obese']
    )
df_clean=pd.get_dummies(df_clean,columns= ['bmi_catagory'],drop_first=True)
df_clean=df_clean.astype(int)
cols=['age','bmi','children']
scaler=StandardScaler()
df_clean[cols]=scaler.fit_transform(df_clean[cols])

#print(df_clean.head())
selected_features=['age','sex','bmi',
                   'children','smoker','region_northwest',
                   'region_southeast','region_southwest',
                   'bmi_catagory_normal','bmi_catagory_overweight',
                   'bmi_catagory_obese']
correlation={
    feature: pearesonr(df_clean[feature], df_clean['charges'])[0] 
    for feature in selected_features
}
correlation_df=pd.DataFrame(list(correlation.items()),
                            columns=['Feature', 'Pearson Correlation'])
correlation_df.sort_values(by='Pearson Correlation', 
                           ascending=False)
#print(correlation_df)
cat_features=['sex','smoker','region_northwest',
              'region_southeast','region_southwest',
                'bmi_catagory_normal','bmi_catagory_overweight',
                'bmi_catagory_obese']
from scipy.stats import chi2_contingency
from scipy.stats import chi2
alpha=0.05
df_clean['charges_catagory']=pd.qcut(
    df_clean['charges'],q=4,labels=False)
chi2_results={}

for col in cat_features:
    contingency_table=pd.crosstab(df_clean[col],df_clean['charges_catagory'])
    chi2_stat, p,_,_=chi2_contingency(contingency_table)
    decision='Reject Null Hypothesis' if p<alpha else 'Accept Null (Drop Features)'
   

    chi2_results[col]={
       'chi2_statistic': chi2_stat,
         'p_value':p,
         'Decision':decision
    }
chi2_df=pd.DataFrame(chi2_results).T
chi2_df=chi2_df.sort_values(by='p_value',ascending=True) 
#df_clean=df_clean.astype(int) 
#print(df_clean.head())
df_final=df_clean.drop(columns=['bmi_catagory_normal',
                                'bmi_catagory_overweight',
                                'region_northwest',
                                'region_southwest',
                                'charges_catagory'])
#print(df_final.head(10))
df_final.to_csv('insurance_cleaned.csv',index=False)
#print('Data Cleaning and Preprocessing Completed. Cleaned data saved to insurance_cleaned.csv')
