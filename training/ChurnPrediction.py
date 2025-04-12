import mlflow
mlflow.set_experiment('Churn_Prediction_Models')

#%% Library import

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import chi2

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

from imblearn.over_sampling import SMOTE

import shap

#%% Load & Prepare Data

df = pd.read_csv('../data/telco-customer-churn.csv')
df = df.drop(['customerID'], axis=1)

print(df.isnull().sum())

print(df.dtypes)

categorical_cols = df.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop(['TotalCharges'])

lb = LabelEncoder()
for col in categorical_cols:
    df[col] = lb.fit_transform(df[col])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

print(df.dtypes)

#%% Vei gorsellestirme

# df['Churn'].value_counts().plot(kind='bar')
# plt.title('Churn Dagilimi')
# plt.show()

# for col in categorical_cols:
#     x = pd.crosstab(df[col], df['Churn'])
#     x.plot(kind='bar', stacked=False)
#     plt.title(col+'Göre Churn Eden Kişi Sayısı')
#     plt.xlabel(col)
#     plt.ylabel('Kişi Sayısı')
#     plt.xticks(rotation=0)
#     plt.show()
    
# numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# for col in numeric_cols:
#     plt.figure()
#     sns.boxplot(x='Churn',y=col,data=df)
#     plt.title(col+' Dagilimi - Churn Karsilastirmasi')
#     plt.show()
    
# for col in numeric_cols:
#     plt.figure()
#     sns.histplot(df,x=col, kde=True, hue='Churn', multiple='stack')
#     plt.title(col+'Dagilimi - Churn')
#     plt.show()
    
# correlation = df[numeric_cols].corr()
# plt.figure(figsize=(6, 4))
# ax = sns.heatmap(correlation, annot=False, cmap='coolwarm')
# for i in range(correlation.shape[0]):
#     for j in range(correlation.shape[1]):
#         text = f"{correlation.iloc[i, j]:.2f}"
#         ax.text(j + 0.5, i + 0.5, text,
#                 ha='center', va='center',
#                 color='black', fontsize=12)

# plt.title('Sayısal Değişkenler Korelasyonu')
# plt.tight_layout()
# plt.show()


# correlation_matrix = df.corr()
# correlation_with_churn = correlation_matrix['Churn'].sort_values(ascending=False)
# print(correlation_with_churn)
# print()
# chi_stats, p_values = chi2(df[categorical_cols], df['Churn'])
# chi2_results = pd.Series(p_values, index=df[categorical_cols].columns).sort_values()
# print(chi2_results)

#%% Drop

drop_cols = [
    "gender", "PhoneService", "TotalCharges"
]
df = df.drop(columns=drop_cols)

#%% Train/Test Split + Scaling

x = df.drop(['Churn'], axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#%% Model Karsilastirmasi

# svc=SVC()
# knn=KNeighborsClassifier(n_neighbors=20)
# dt=DecisionTreeClassifier()
# rf=RandomForestClassifier()
# ada=AdaBoostClassifier()
# lr=LogisticRegression()
# gbc=GradientBoostingClassifier()
# xgb=XGBClassifier()
# v1=VotingClassifier(estimators=[('svc', svc),('knn', knn),('dt', dt),('rf', rf),('ada', ada),('lr',lr),('gbc',gbc),('xgb',xgb)])

# names = ['SVC','KNN','Desicion Tree','Random Forest','AdaBoost','Logistic Regression','Gradient BC','XGB','Voting']
# classifiers = [svc, knn, dt, rf, ada, lr, gbc, xgb,v1]

# for name, clf in zip(names, classifiers):
#     clf.fit(x_train_scaled, y_train)
#     score = clf.score(x_test_scaled, y_test)
#     print('{}: test score: {}'.format(name, score))
#     score_train = clf.score(x_train_scaled, y_train)
#     print('{}: train score: {}'.format(name, score_train))
    
    
#%% Logistic Regression + RFE + GridSearchCV
with mlflow.start_run():
    try:
        smote = SMOTE(random_state=42)
        x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

        modelLR = LogisticRegression()
        rfe = RFE(modelLR, n_features_to_select=15)
        rfe.fit(x_train_smote, y_train_smote)
        
        selected_features = x_train.columns[rfe.support_]
        x_train_rfe = pd.DataFrame(rfe.transform(x_train_smote), columns=selected_features)
        x_test_rfe = pd.DataFrame(rfe.transform(x_test), columns=selected_features)

        scaler_rfe = MinMaxScaler()
        x_train_rfe_scaled = scaler_rfe.fit_transform(x_train_rfe)
        x_test_rfe_scaled = scaler_rfe.transform(x_test_rfe)
        params_lr = [
            {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'newton-cg']},
            {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'saga']}
        ]
        
        grid_lr = GridSearchCV(LogisticRegression(), params_lr, cv=5, scoring='accuracy')
        grid_lr.fit(x_train_rfe_scaled, y_train_smote)
        
        best_params = grid_lr.best_params_
        mlflow.log_params(best_params)
        mlflow.log_param("penalty", best_params["penalty"])
        mlflow.log_param("C", best_params["C"])
        mlflow.log_param("solver", best_params["solver"])

        y_pred_lr = grid_lr.predict(x_test_rfe_scaled)
        y_proba_lr = grid_lr.predict_proba(x_test_rfe_scaled)[:,1]
        
        acc = accuracy_score(y_test, y_pred_lr)
        roc_score = roc_auc_score(y_test, y_proba_lr)
    
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_score)
    
        # model loglama + input örneği
        mlflow.sklearn.log_model(
            sk_model=grid_lr.best_estimator_,
            artifact_path="model",
            input_example=x_test_rfe_scaled.iloc[:1]
        )
        print("MLflow loglaması tamamlandı.")
    except Exception as e:
        print('Hata var ', e)

print('Confusion Matrix LR: ')
print(confusion_matrix(y_test, y_pred_lr))

print('Classification Report LR: ')
print(classification_report(y_test, y_pred_lr))

print('ROC AUC Score LR: ', roc_score)
print('------------------------------')
# fpr, tpr,_=roc_curve(y_test, y_proba_lr)
# plt.plot(fpr, tpr, label=f'ROC AUC = {roc_score: .2f}')
# plt.plot([0,1],[0,1],'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC CURVE LR')
# plt.legend()
# plt.show()


#%% SHAP Explainability
 
# explainer = shap.Explainer(grid_lr.best_estimator_, x_train_rfe)
# shap_values = explainer(x_test_rfe)
# shap.plots.beeswarm(shap_values)

#%%

import joblib
joblib.dump(grid_lr.best_estimator_, "../backend/model.pkl")

joblib.dump(list(selected_features), '../backend/selected_features.pkl')
joblib.dump(scaler_rfe, '../backend/scaler.pkl')


