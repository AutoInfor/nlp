# -*- coding: utf-8 -*-
# @Time         : 2018-07-28 23:31
# @Author       : Jayce Wong
# @ProjectName  : NLP
# @FileName     : classifier.py
# @Blog         : http://blog.51cto.com/jayce1111
# @Github       : https://github.com/SysuJayce


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

from feature_extractors import bow_extractor,tfidf_extractor,word2vec_extractor
from get_data import get_data,corpus2token

   

def train_fixed_parameter_model(extractor_method,key, train_text, train_result,                                 test_text, test_result):
    classifier=models[key]
    #kfold = StratifiedKFold(n_splits=5,shuffle=False,random_state=0)
    kfold=KFold(n_splits=num_folds,random_state=seed)
    accuracy = cross_val_score(classifier, 
        train_text, train_result, scoring='accuracy', cv=kfold)
    
    precision = cross_val_score(classifier, 
        train_text, train_result, scoring='precision_weighted', cv=kfold)
    recall = cross_val_score(classifier, 
        train_text, train_result, scoring='recall_weighted', cv=kfold)
    f1 = cross_val_score(classifier, 
        train_text, train_result, scoring='f1_weighted', cv=kfold)
    
    print("训练数据：Accuracy:{}  Precision:{}  Recall:{}  F1:{}".format(str(round(accuracy.mean(), 2)),str(round(precision.mean(), 2)),str(round(recall.mean(), 2)),str(round(f1.mean(), 2))))
    
    classifier.fit(train_text,train_result)
    predictions = classifier.predict(test_text)
    true_labels=test_result
    predicted_labels=predictions
    
    print("测试数据：Accuracy:{}  Precision:{}  Recall:{}  F1:{}".format(str(round(metrics.accuracy_score(true_labels,predicted_labels), 2)),str(round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2))))   
    
        
    score_list.append([str(round(accuracy.mean(), 2)),str(round(precision.mean(), 2)),str(round(recall.mean(), 2)),str(round(f1.mean(), 2)),key+"_val",extractor_method])
    results.append(accuracy)
    score_list.append([str(round(metrics.accuracy_score(true_labels,predicted_labels), 2)),str(round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2)),key+"_test",extractor_method])
    results.append([metrics.accuracy_score(true_labels,predicted_labels)])
    

def train_search_parameter_model(classifier, train_text, train_result,
                                 test_text, test_result):

    print('train_text[:20]',train_text[:20],train_result[:20])

    grid_result=classifier.fit(train_text,train_result)
    print('最优参数:%s,获取分数:%s'%(grid_result.best_params_,grid_result.best_score_))
    cv_results=zip(grid_result.cv_results_['mean_test_score'],
               grid_result.cv_results_['std_test_score'],
               grid_result.cv_results_['params'])

    for mean,std,param in cv_results:
        print('%f (%f) with %r'%(mean,std,param))
        
    predictions = classifier.predict(test_text)
    true_labels=test_result
    predicted_labels=predictions
    print("测试数据：Accuracy:{}  Precision:{}  Recall:{}  F1:{}".format(str(round(metrics.accuracy_score(true_labels,predicted_labels), 2)),str(round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 2)),str(round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2))))

    # # 显示混淆矩阵
    print("\n混淆矩阵\n",pd.crosstab(test_result, predictions, rownames = ['label'], colnames = ['predict']))

    test_df['预测结果'] = label_encoder.inverse_transform(predictions)
    print('打印test_df预测结果\n',test_df)
    test_df.to_excel("output_"+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+".xlsx",engine='xlsxwriter')
    
def best_param(modelname,extractor_method):
    if extractor_method=='tfidf':
        train_text_token_extractor,test_text_token_extractor=train_text_token_tfidf,test_text_token_tfidf
    if extractor_method=='word2vec':
        train_text_token_extractor,test_text_token_extractor=train_text_token_word2vec,test_text_token_word2vec
    if extractor_method=='bow':
        train_text_token_extractor,test_text_token_extractor=train_text_token_bow,test_text_token_bow      
    
    if modelname=='KNN':
        # 调参KNN
        param_grid={'n_neighbors':[1,3,5,7,9,11,13,15,17,19,21]}
        model=KNeighborsClassifier()
    if modelname=='SVC':
        # 调参SVC
        param_grid={}
        param_grid['C']={0.1,0.3,0.5,0.7,0.9,1.0,1.3,1.5,1.7,2.0}
        param_grid['kernel']={'linear','poly','rbf','sigmoid','precomputed'}
        model=SVC()
    if modelname=='LR':
        # 调参LR
        param_grid = {}
        param_grid['C'] = [0.1, 5, 13, 15]
        model = LogisticRegression()
    if modelname=='SGD':
        # 调参LR
        param_grid = {}
        param_grid['loss'] = ["hinge", "log"]
        param_grid['penalty'] = ['l2', 'l1', 'elasticnet']
        model = SGDClassifier()
    if modelname=='MNB':
        # 调参MNB
        param_grid = {}
        param_grid['alpha'] = [0.001, 0.01, 0.1, 1.5]
        model = MultinomialNB()
    if modelname=='RF':
        # 调参RF
        param_grid = {}
        param_grid['n_estimators'] = [10, 100, 150, 200]
        model = RandomForestClassifier()
    if modelname=='CART':
        # 调参RF
        param_grid = {}
        param_grid['criterion'] = ["gini","entropy"]
        model = DecisionTreeClassifier()
    #10折交叉验证
    kfold=KFold(n_splits=num_folds,random_state=seed)

    #网格搜索
    grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
    
    train_search_parameter_model(classifier=grid,train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    
#读取数据
train_text_token,train_result_label_encoder,train_df,test_text_token,test_result_label_encoder,test_df,label_encoder=get_data()
print('\ntrain_df内容\n',train_df)

# 词袋模型
bow_vectorizer, train_text_token_bow = bow_extractor(train_text_token)
test_text_token_bow = bow_vectorizer.transform(test_text_token)

# tfidf模型
tfidf_vectorizer, train_text_token_tfidf = tfidf_extractor(train_text_token)
test_text_token_tfidf = tfidf_vectorizer.transform(test_text_token)

# Word2Vec模型
train_text_token_word2vec=word2vec_extractor(train_text_token)
test_text_token_word2vec=word2vec_extractor(test_text_token)

models={}
models['LR']=LogisticRegression()
models['SGD']=SGDClassifier()
models['MNB']=MultinomialNB()
models['GNB']=GaussianNB()
models['KNN']=KNeighborsClassifier()
models['SVM']=SVC()
models['RFC']=RandomForestClassifier()
models['CART']=DecisionTreeClassifier()
#models['LDA']=LinearDiscriminantAnalysis()
#比较各算法的准确度
num_folds=10
seed=7
scoring='accuracy'

results=[]
score_list=[]
for key in list(models.keys())[:1]:
    for train_text_token_extractor,test_text_token_extractor,extractor_method in [(train_text_token_tfidf,test_text_token_tfidf,'tfidf'),(train_text_token_word2vec,test_text_token_word2vec,'word2vec'),(train_text_token_bow,test_text_token_bow,'bow')]:
        print("\n采用模型为：",key,' 文本编码为：'+extractor_method)
        try:
            train_fixed_parameter_model(extractor_method,key,train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
        except Exception as e:
            print(e)
        
df = pd.DataFrame(score_list, columns=['Accuracy','Precision','Recall','F1','算法模型','文本编码',])
print(df.sort_values(by='Accuracy',ascending=False))

#评估算法---箱线图
fig=plt.figure()
fig.suptitle('algorithm comparison')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(df['算法模型']+df['文本编码'])


scores,model,extractor_method=df.sort_values(by='Accuracy',ascending=False).iloc[0,[0,-2,-1]]
print('scoresMAX ',scores,'model ',model.split('_')[0],'extractor_method ',extractor_method)
  
best_param(model.split('_')[0],extractor_method)

plt.show()
























'''

for train_text_token_extractor,test_text_token_extractor,extractor_method in [(train_text_token_word2vec,test_text_token_extractor,'Word2Vec模型'),(train_text_token_bow,test_text_token_bow,'词袋模型'),(train_text_token_tfidf,test_text_token_tfidf,'tfidf模型')]:
    strKFold = StratifiedKFold(n_splits=5,shuffle=False,random_state=0)
    print('\n****************'+extractor_method+"****************"+"\ntrain_text_token_extractor[:3]\n",train_text_token_extractor[:3])

    print("\nNavie Bayes ")
    mnb = MultinomialNB(alpha=0.0001)
    if extractor_method=='Word2Vec模型':
        mnb = GaussianNB()#特征值非负
    
    train_fixed_parameter_model(
        classifier=mnb,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    mnb_parameter_grid = [{'alpha': [1, 0.01, 0.0001], 'fit_prior': ['false', 'true']}]
    mnb = GridSearchCV(MultinomialNB(),mnb_parameter_grid, cv=strKFold, scoring='recall_weighted') 
    train_search_parameter_model(
        classifier=mnb,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)

    print("\nLogistic Regression ")
    lr = LogisticRegression(C=10000)
    lr_tfidf_predictions = train_fixed_parameter_model(
        classifier=lr,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    lr_parameter_grid = [{'C': [0.1, 5, 100,10000],'solver': ['newton-cg','lbfgs', 'liblinear', 'sag']}]
    lr = GridSearchCV(LogisticRegression(),lr_parameter_grid, cv=strKFold, scoring='recall_weighted') 
    train_search_parameter_model(
        classifier=lr,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)    

    print("\nSGD ")
    sgd = SGDClassifier()
    svm_tfidf_predictions = train_fixed_parameter_model(
        classifier=sgd,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    sgd_parameter_grid = [{'loss': ["hinge", "log"], 'penalty': ['l2', 'l1', 'elasticnet']}]
    sgd = GridSearchCV(SGDClassifier(),sgd_parameter_grid, cv=strKFold, scoring='recall_weighted')
    train_search_parameter_model(
        classifier=sgd,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)

    print("\nRandomForest ")
    rf=RandomForestClassifier(n_estimators=200, max_depth=8, random_state=7)
    rf_tfidf_predictions = train_fixed_parameter_model(
        classifier=rf,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    rf_parameter_grid = [{'n_estimators':[10, 200, 600], 'max_depth':[2, 8, 20], 'random_state': [1, 7, 20]}]
    rf = GridSearchCV(RandomForestClassifier(class_weight='balanced'),rf_parameter_grid, cv=5, scoring='recall_weighted')        
    train_search_parameter_model(
        classifier=rf,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)  

    print("\nSVC ")
    svc = SVC(C=0.001)#C=1.0,class_weight='auto'
    svc_tfidf_predictions = train_fixed_parameter_model(classifier=svc,train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
    svc_parameter_grid = [{'kernel': ['linear'], 'C': [1, 600]},{'kernel': ['rbf'], 'C': [1, 600]}]
    #svc_parameter_grid = [{'kernel': ['linear'], 'C': [1, 600]},{'kernel': ['poly'], 'degree': [2, 3]},{'kernel': ['rbf'], 'gamma': [0.01, 0.001], 'C': [1, 600]}]
    svc = GridSearchCV(SVC(),svc_parameter_grid, cv=strKFold, scoring='recall_weighted')
    train_search_parameter_model(
        classifier=svc,
        train_text=train_text_token_extractor, train_result=train_result_label_encoder,test_text=test_text_token_extractor, test_result=test_result_label_encoder)
      



'''
