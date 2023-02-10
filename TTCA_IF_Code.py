#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from collections import Counter

def evaluate_model_test(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict_proba(X_test)[:,1]
    for i in range(len(y_pred)):
        if y_pred[i]>0.5:
            y_pred[i]=1
        else:
            y_pred[i]=0
    

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    #MCC
    mcc=matthews_corrcoef(y_test, y_pred)
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    total=sum(sum(cm))
    
    #accuracy=(cm[0,0]+cm[1,1])/total
    spec = cm[0,0]/(cm[0,1]+cm[0,0])
    sen= cm[1,1]/(cm[1,0]+cm[1,1])

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'mcc':mcc,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'sen': sen, 'spec':spec}

def evaluate_model_train(model, X_train, y_train):
    from sklearn import metrics
    conf_matrix_list_of_arrays = []
    mcc_array=[]
    #cv = KFold(n_splits=5)
    #cv = StratifiedKFold(n_splits=5)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
    lst_accu = []
    AUC_list=[]
    prec_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='precision'))
    recall_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='recall'))
    f1_train=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='f1'))
    Acc=np.mean(cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy'))
    for train_index, test_index in cv.split(X_train, y_train): 
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index] 
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index] 
        model.fit(X_train_fold, y_train_fold) 
        lst_accu.append(model.score(X_test_fold, y_test_fold))
        acc=np.mean(lst_accu)
        
        conf_matrix = confusion_matrix(y_test_fold, model.predict(X_test_fold))
        conf_matrix_list_of_arrays.append(conf_matrix)
        cm = np.mean(conf_matrix_list_of_arrays, axis=0)
        mcc_array.append(matthews_corrcoef(y_test_fold, model.predict(X_test_fold)))
        mcc=np.mean(mcc_array, axis=0)
        
        AUC=metrics.roc_auc_score( y_test_fold, model.predict_proba(X_test_fold)[:,1])
        AUC_list.append(AUC)
        auc=np.mean(AUC_list)
        
        
    total=sum(sum(cm))
    accuracy=(cm[0,0]+cm[1,1])/total
    specificity = cm[0,0]/(cm[0,1]+cm[0,0])
    sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
       
    
    return {'prec_train': prec_train, 'recall_train': recall_train, 'f1_train': f1_train, 'cm': cm, 'mcc': mcc,'Acc':Acc,
           'sen':sensitivity,'spec':specificity, 'acc':acc, 'lst_accu':lst_accu, 'AUC':auc}

# Import Data
data= pd.read_csv(r"C:/Users/MirTanveer/Desktop/iTTCA/Dataframe_/Optuna_PP_150_Training.csv")
data1= pd.read_csv(r"C:/Users/MirTanveer/Desktop/iTTCA/Dataframe_/Optuna_PP_150_Testing.csv")

X_train = data.iloc[:,1:-1].values
y_train = data.iloc[:,-1].values

X_test = data1.iloc[:,1:-1].values
y_test = data1.iloc[:,-1].values

X_train.shape, X_test.shape

# transform the dataset
import imblearn
from imblearn.over_sampling import SMOTE,  ADASYN
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
oversample = SMOTEENN()
X_train, y_train = oversample.fit_resample(X_train, y_train)
print(sorted(Counter(y_train).items()))

cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1)

#Random Forest
import optuna
from sklearn.ensemble import RandomForestClassifier
def RF_objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int('max_depth', 1, 60)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    min_samples_split= trial.suggest_int("min_samples_split", 2, 20)
    max_features= trial.suggest_categorical('max_features', ['auto','sqrt','log2'])
    ## Create Model
    model = RandomForestClassifier(max_depth = max_depth, min_samples_split=min_samples_split,
                                   n_estimators = n_estimators,n_jobs=2,
                                   max_features=max_features, random_state=25)


    model.fit(X_train,y_train)    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean

#Execute optuna and set hyperparameters
RF_study = optuna.create_study(direction='maximize')
RF_study.optimize(RF_objective, n_trials=100)

optimized_RF=RandomForestClassifier(**RF_study.best_params)

# Evaluate Model on Training data
train_eval = evaluate_model_train(optimized_RF, X_train, y_train)
print("Confusion Matrix is: ", train_eval['cm'])
print ('Accuracy : ', train_eval['Acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Mean of Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("The Acc value from CM is: ", train_eval['acc'])
print("The Recall value is: ", train_eval['recall_train'])
print("The F1 score is: ", train_eval['f1_train'])
print('The area under curve is:', train_eval['AUC'])
#print('5 accuracies: ', train_eval['lst_accu'])

# Evaluate Model on Testing data
#rfc.fit(X_train, y_train)
dtc_eval = evaluate_model_test(optimized_RF, X_test, y_test)
# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# tsne1=rfc.predict_proba(X_train)[:,1]
# tsne1=pd.DataFrame(tsne1)

# ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
            'n_estimators' : trial.suggest_int('n_estimators', 100, 1000),
            'max_depth' : trial.suggest_int('max_depth', 10, 60),
            'max_leaf_nodes' : trial.suggest_int('max_leaf_nodes', 15, 77),
            'criterion' : trial.suggest_categorical('criterion', ['gini', 'entropy'])

    }


    # Fit the model
    etc_model = ExtraTreesClassifier(**params)
    etc_model.fit(X_train, y_train)
    score = cross_val_score(etc_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
etc_study = optuna.create_study(direction='maximize')
etc_study.optimize(objective, n_trials=100)

optimized_etc =ExtraTreesClassifier(**etc_study.best_params)

# Evaluate Model on Training data
train_eval = evaluate_model_train(optimized_etc, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['Acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print('The area under curve is:', train_eval['AUC'])
print("F1 score is: ", train_eval['f1_train'])

# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(optimized_etc, X_test, y_test)
# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# tsne2=etc.predict_proba(X_train)[:,1]
# tsne2=pd.DataFrame(tsne2)

# CatBoost
from catboost import CatBoostClassifier
import optuna
def objective(trial):
    """Define the objective function"""
    param = {
        "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        
        
    }


    # Fit the model
    model = CatBoostClassifier(**param)
    model.fit(X_train, y_train)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

optimized_cbc =CatBoostClassifier(**study.best_params)

# Evaluate Model on Training data
train_eval = evaluate_model_train(optimized_cbc, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print('The area under curve is:', train_eval['AUC'])
print("F1 score is: ", train_eval['f1_train'])

# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(optimized_cbc, X_test, y_test)
# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# tsne3=CBC2.predict_proba(X_train)[:,1]
# tsne3=pd.DataFrame(tsne3)

# XGB

from xgboost import XGBClassifier
#cv = RepeatedStratifiedKFold(n_splits=5)
import optuna
def objective(trial):
    """Define the objective function"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 370),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 10.0),
        'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
        #'eval_metric': 'mlogloss',
        #'use_label_encoder': False
    }

    # Fit the model
    xgb_model = XGBClassifier(**params, use_label_encoder =False, eval_metric='mlogloss')
    xgb_model.fit(X_train, y_train)
    cv = RepeatedStratifiedKFold(n_splits=5)
    score = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()
    return accuracy_mean
#Execute optuna and set hyperparameters
XGB_study = optuna.create_study(direction='maximize')
XGB_study.optimize(objective, n_trials=100)
optimized_XGB =XGBClassifier(**XGB_study.best_params, use_label_encoder=False)

# Evaluate Model on Training data
train_eval = evaluate_model_train(optimized_XGB, X_train, y_train)

print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])
print('The area under curve is:', train_eval['AUC'])

# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(optimized_XGB, X_test, y_test)

# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# tsne4=xgb.predict_proba(X_train)[:,1]
# tsne4=pd.DataFrame(tsne4)

# LGBM
import lightgbm as lgbm
import optuna
def objective(trial):
    """Define the objective function"""
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 50, step=1, log=False), 
        'max_depth': trial.suggest_int('max_depth', 1, 30, step=1, log=False), 
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 1, log=True), 
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=25), 
        #'objective': 'multiclass', 
       # 'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50, log=False), 
        #'subsample': trial.suggest_uniform('subsample', 0.7, 1.0), 
        #'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
        #'reg_alpha': trial.suggest_uniform('reg_alpha', 0.0, 1.0),
        #'reg_lambda': trial.suggest_uniform('reg_lambda', 0.0, 10.0),
        'random_state': 0
    }


    # Fit the model
    lgbm_model = lgbm.LGBMClassifier(**params)
    lgbm_model.fit(X_train, y_train, verbose=False)
    score = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring="accuracy")
    accuracy_mean = score.mean()

    return accuracy_mean


#Execute optuna and set hyperparameters
lgbm_study = optuna.create_study(direction='maximize')
lgbm_study.optimize(objective, n_trials=100)

optimized_lgbm =lgbm.LGBMClassifier(**lgbm_study.best_params)

# Evaluate Model on Training data
train_eval = evaluate_model_train(optimized_lgbm, X_train, y_train)
print("Confusion Matrix is: ", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Mean of Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("The Precision value is: ", train_eval['prec_train'])
print("The Recall value is: ", train_eval['recall_train'])
print("The F1 score is: ", train_eval['f1_train'])
print('The area under curve is:', train_eval['AUC'])

# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(optimized_lgbm, X_test, y_test)
# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

#tsne5D=pd.concat([tsne1, tsne2, tsne3, tsne4, tsne5], axis=1)

# VotingClassifier

from sklearn.ensemble import VotingClassifier
vclf = VotingClassifier(estimators=[ ('RF', optimized_RF),  ('XGB', optimized_XGB), ("CBC", optimized_cbc), ('ETC', optimized_etc), ('LGBM', optimized_lgbm)], voting='soft')
# vclf.fit(X_train, y_train)
# y_pred=vclf.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)

#Evaluate Model on Training data
train_eval = evaluate_model_train(vclf, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])
print('The area under curve is:', train_eval['AUC'])

# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(vclf, X_test, y_test)
# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

# Meta-Classifier

# defining meta-classifier
from mlxtend.classifier import StackingClassifier
clf_stack = StackingClassifier(classifiers =[optimized_cbc, optimized_RF, optimized_XGB,optimized_lgbm], 
                               meta_classifier = optimized_etc, use_probas = True, use_features_in_secondary = True)

# Evaluate Model on Training data
train_eval = evaluate_model_train(clf_stack, X_train, y_train)

print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])
print('The area under curve is:', train_eval['AUC'])

#Evaluate Model on Testing data
dtc_eval = evaluate_model_test(clf_stack, X_test, y_test)

# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

#Ensemble 

from sklearn import model_selection
from mlxtend.classifier import EnsembleVoteClassifier
en_clf = EnsembleVoteClassifier(clfs=[optimized_XGB, optimized_RF, optimized_cbc, optimized_lgbm, optimized_etc], weights=[1,1,1,1,1])
#eclf.fit(X_train, y_train)
#y_predd=eclf.predict(X_test)

#Evaluate Model on Training data
train_eval = evaluate_model_train(en_clf, X_train, y_train)

print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])

#Evaluate Model on Testing data
dtc_eval = evaluate_model_test(en_clf, X_test, y_test)

# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

#testing on independent dataset

#import pickle
#pickled_model = pickle.load(open('Trained_Models/Ensemble_Model.pkl', 'rb'))
#y_pred=pickled_model.predict(X_test)

#Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(cm)
#accuracy_score(y_test, y_pred)
