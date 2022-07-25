#%% Import libraries
import os
import pickle
import pandas as pd

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from Module import displot_graph,countplot_graph,boxplot,confusion_mat,LogisticReg
#%% Loading data
CSV_PATH = os.path.join(os.getcwd(),'Data', 'heart.csv')
df = pd.read_csv(CSV_PATH)

# Checking data
df
df.describe().T
df.info()
# %%
# Constant
con_col = ['age', 'trtbps', 'chol','thalachh','oldpeak']
cat_col = ['sex', 'cp', 'fbs', 'restecg', 'exng','slp','caa','thall','output']

#%% Data visualization
displot_graph(con_col,df)
# oldpeak have outlier,chol
countplot_graph(cat_col,df)

# display unique value percentages for target
item_counts = df["output"].value_counts(normalize=True)
print(item_counts)
# 'ok' balanced dataset not higher than 30% difference

# caa should be '0-3' but in data have extra '4' category, change to NaNs
# box plot (to see outliers more clearly)
boxplot(df=df,con_col=con_col,nrows=5,ncols=1,size1=(30,40))
# Outliers in con_col
# The dataset is very small, not worth it to remove outlier

#%% Data cleaning
# check NaNs percentage
df.isna().sum()/len(df)*100
# caa need to change 4 to NaNs and thall 0 is NaNs

# Check duplicates
df.duplicated().sum()
df[df.duplicated()]

# drop 1 duplicated row
df = df.drop_duplicates()
df.duplicated().sum()

# Handling thall and caa columns
df['caa'].value_counts()

df['thall'].value_counts()


# only 5 from caa and 2 from thall to drop
df = df.loc[(df['thall']!=0) & (df['caa']!=4)]
df.describe().T

# Checking NaNs again
df.isna().sum()/len(df)*100

# %% Features selection
# Define X,y
X = df.drop(labels='output',axis=1)
y = df['output']

cat_col.remove('output')
# cramers V for cat vs cat d ata
confusion_mat(X,y,cat_col)


# logistic regression for cat vs con data
LogisticReg(X,y,con_col)
# good correlation from con_col most higher than 0.5

# %% Data preprocessing
X = df.drop(labels='output',axis=1)
y= df['output']
 
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                    test_size=0.3,random_state=123)

#%% Machine learning model
# Finding the best model
# Pipeline
# Logistic regression
pipeline_mms_lr = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('Logistic_Classifier', LogisticRegression())
                        ])

pipeline_ss_lr = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('Logistic_Classifier', LogisticRegression())])

# Decision tree
pipeline_mms_dt = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('DecisionTreeClassifier', DecisionTreeClassifier())
                        ])

pipeline_ss_dt = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('DecisionTreeClassifier', DecisionTreeClassifier())])
# Random forest
pipeline_mms_rf = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('RFC', RandomForestClassifier())
                        ])

pipeline_ss_rf = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('RFC', RandomForestClassifier())])

# SVM
pipeline_mms_svm = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('SVC', SVC())
                        ])

pipeline_ss_svm = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('SVC', SVC())])
                        
# KNN
pipeline_mms_KNN = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('KNN', KNeighborsClassifier())
                        ])

pipeline_ss_KNN = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('KNN', KNeighborsClassifier())])

# GBC
pipeline_mms_GBC = Pipeline([
                        ('Min_Max_Scaler', MinMaxScaler()),
                        ('GBC', GradientBoostingClassifier())
                        ])

pipeline_ss_GBC = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('GBC', GradientBoostingClassifier())])

# Create a list to store all the pipeline
pipelines = [pipeline_mms_lr, pipeline_ss_lr, pipeline_mms_dt, pipeline_ss_dt,
             pipeline_mms_rf, pipeline_ss_rf, pipeline_mms_svm, pipeline_ss_svm,
             pipeline_mms_KNN, pipeline_ss_KNN, pipeline_mms_GBC, pipeline_ss_GBC]

for pipe in pipelines:
    pipe.fit(X_train,y_train)

# Create a dictionary for pipeline types
pipe_dict = {0: 'MMS+LogReg', 1:'SS+LogReg', 2:'MMS+DecTree', 3:'SS+DecTree',
            4:'MMS+RanForest', 5:'SS+RanForest', 6:'MMS+SVM', 7:'SS+SVM',
            8:'MMS+KNN', 9:'SS+KNN',10:'MMS+GBC',11:'SS+GBC'}

best_score = 0
for i, pipe in enumerate(pipelines):
    print(pipe_dict[i])
    print(pipe.score(X_test,y_test))
    if pipe.score(X_test,y_test) > best_score:
        best_score = pipe.score(X_test,y_test)
        best_pipeline = pipe

print(f'The best scaler and classifer for this Data is {best_pipeline.steps} with score of {best_score}')
 # %%
# Model Analysis
y_pred = best_pipeline.predict(X_test)
y_true = y_test


cr = classification_report(y_true,y_pred)

print(cr)
# %% GridsearchCV
# So best pipeline is for standard scaler and logistic regression model
# Hence using GridSearchCV to find the best paramaters for the model
pipeline_ss_rf = Pipeline([
                        ('Standard_Scaler', StandardScaler()),
                        ('RFC', RandomForestClassifier())])

grid_param = [{'RFC__n_estimators': [25, 50, 100, 200, 400]}] #Hyperparameters

grid_search = GridSearchCV(pipeline_ss_rf, param_grid=grid_param,cv=5,
            verbose=1, n_jobs=-1)

grid = grid_search.fit(X_train,y_train)
print(grid.score(X_test,y_test))
print(grid.best_index_)
print(grid.best_params_)
best_model = grid.best_estimator_
# %%
BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(), 'Models', 'best_estimator.pkl')

with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(best_model,file)

# Model Analysis
y_pred = best_model.predict(X_test)
y_true = y_test

# cm = confusion_matrix(y_true,y_pred, normalize='all')
cr = classification_report(y_true,y_pred)

print(cr)