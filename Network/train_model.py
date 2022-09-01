#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


class BetaEncoder(object):
        
    def __init__(self, group):
        
        self.group = group
        self.stats = None
        
    # get counts from df
    def fit(self, df, target_col):
        self.prior_mean = np.mean(df[target_col])
        stats = df[[target_col, self.group]].groupby(self.group)
        stats = stats.agg(['sum', 'count'])[target_col]    
        stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
        stats.reset_index(level=0, inplace=True)           
        self.stats = stats
        
    # extract posterior statistics
    def transform(self, df, stat_type, N_min=1):
        
        df_stats = pd.merge(df[[self.group]], self.stats, how='left')
        n = df_stats['n'].copy()
        N = df_stats['N'].copy()
        
        # fill in missing
        nan_indexs = np.isnan(n)
        n[nan_indexs] = self.prior_mean
        N[nan_indexs] = 1.0
        
        # prior parameters
        N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean*N_prior
        beta_prior = (1-self.prior_mean)*N_prior
        
        # posterior parameters
        alpha = alpha_prior + n
        beta =  beta_prior + N-n
        
        # calculate statistics
        if stat_type=='mean':
            num = alpha
            dem = alpha+beta
                    
        elif stat_type=='mode':
            num = alpha-1
            dem = alpha+beta-2
            
        elif stat_type=='median':
            num = alpha-1/3
            dem = alpha+beta-2/3
        
        elif stat_type=='var':
            num = alpha*beta
            dem = (alpha+beta)**2*(alpha+beta+1)
                    
        elif stat_type=='skewness':
            num = 2*(beta-alpha)*np.sqrt(alpha+beta+1)
            dem = (alpha+beta+2)*np.sqrt(alpha*beta)

        elif stat_type=='kurtosis':
            num = 6*(alpha-beta)**2*(alpha+beta+1) - alpha*beta*(alpha+beta+2)
            dem = alpha*beta*(alpha+beta+2)*(alpha+beta+3)
            
        # replace missing
        value = num/dem
        value[np.isnan(value)] = np.nanmedian(value)
        return value
def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()




print('load data!')
cat_cols = ['location',
       'employment_type',
       'required_experience' ,
       'required_education', 
       'industry',
       'function']
df = pd.read_csv('./data/TF_IDF.csv',encoding='latin1')
df1 = df.drop(df.columns[[0,1,2,4,5,6,7,8,9,19]],axis=1)
X = df1
y = df1['fraudulent']
X1 = df
y1 = df['fraudulent']


plot_params = ['location','telecommuting',
               'has_company_logo','has_questions',
              'employment_type','required_experience',
              'required_education','industry',
              'function']


for e in plot_params:
    fig,ax = plt.subplots(figsize=(15,15), dpi=100)
    splot = sns.countplot(x = e,
                hue = 'fraudulent',
                data = df,
                order = df[e].value_counts().index) 
    splot.set_xlabel(e,fontsize = 20)
    splot.set_ylabel('count',fontsize = 20)
    plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize=14)
    #在長條圖上顯示數值
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.09, p.get_height()+50), color='black', size=20)
    sfig = splot.get_figure()
    sfig.savefig(f'{e}.png',  orientation="landscape")
 





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=2022, stratify=y)
print('X_train:', X_train.shape)
print('X_test:', X_test.shape)





y1_test.index = np.arange(1, len(y1_test) + 1)
y1_test.to_csv('answer.csv')





X2 = X1_test.iloc[:,1:6]
X2.index = np.arange(1, len(X2) + 1)
X2.to_csv('testdata.csv')





n_rounds = 10000 # increase to 2000



for N_min in [1, 10, 100, 1000,10000]: 

    print('label encoding')
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(np.concatenate([X_train[col], X_test[col]]))
        X_train[col] = le.transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
    scores = []
    feature_cols = []
    print(f'{N_min}')
    for c in cat_cols:
        # fit encoder
        be = BetaEncoder(c)
        be.fit(X_train,'fraudulent')

        # mean
        feature_name = f'{c}_mean'
        X_train[feature_name] = be.transform(X_train, 'mean', N_min)
        X_test[feature_name] = be.transform(X_test, 'mean', N_min) 

        # mode
        feature_name = f'{c}_mode'
        X_train[feature_name] = be.transform(X_train, 'mode', N_min)
        X_test[feature_name] = be.transform(X_test, 'mode', N_min)

        # median
        feature_name = f'{c}_median'
        X_train[feature_name] = be.transform(X_train, 'median', N_min)
        X_test[feature_name] = be.transform(X_test, 'median', N_min)

        # var
        feature_name = f'{c}_var'
        X_train[feature_name] = be.transform(X_train, 'var', N_min)   
        X_test[feature_name] = be.transform(X_test, 'var', N_min)

        # skewness
        feature_name = f'{c}_skewness'
        X_train[feature_name] = be.transform(X_train, 'skewness', N_min)
        X_test[feature_name] = be.transform(X_test, 'skewness', N_min)

        # kurtosis
        feature_name = f'{c}_kurtosis'
        X_train[feature_name] = be.transform(X_train, 'kurtosis', N_min)
        X_test[feature_name] = be.transform(X_test, 'kurtosis', N_min)
        feature_cols.append(feature_name)
                                

            # setup lightgbm data
    X_train1 = X_train.drop(columns=cat_cols,axis=1)
    X_test1 = X_test.drop(columns=cat_cols,axis=1)
    X_train1 = X_train1.drop(columns='fraudulent',axis = 1)
    X_test1 = X_test1.drop(columns='fraudulent',axis=1)
    X_test1.index = np.arange(1, len(X_test1) + 1)
    X_test1.to_csv('test.csv')
    clf = lgb.LGBMClassifier(colsample_bytree=0.45, learning_rate=0.057, is_unbalance =True, max_depth=14,
               min_child_weight=20.0, n_estimators=450, num_leaves=5,
               random_state=1, reg_lambda=2.0, subsample=0.99,
               subsample_freq=6)
    clf.fit(X_train1,y_train)

    clf.booster_.save_model(f'lightgbm_model_{N_min}.txt')
    pred = clf.predict(X_test1)
    print("Acc:",accuracy_score(y_test,pred))
        


fig, ax = plt.subplots(figsize=(10,8))
cm = confusion_matrix(y_test, pred)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
splot = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix\n")
plt.tight_layout()
plt.show()
sfig = splot.get_figure()
sfig.savefig('confusion_Matrix.png',  orientation="landscape")




from sklearn.metrics import classification_report
print(classification_report(y_test, pred))




print('Training set score: {:.4f}'.format(clf.score(X_train1, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test1, y_test)))




from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y_test,pred)
auc = metrics.roc_auc_score(y_test,pred)
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.savefig('ROC.png', bbox_inches='tight')



warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns),reverse=True), columns=['Value','Feature'])
feature_imp = feature_imp.drop(4)
fig,ax = plt.subplots(figsize=(10,10), dpi=100)
splot = sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).head(5))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
sfig = splot.get_figure()
sfig.savefig('lgbm_importances.png',  orientation="landscape")







