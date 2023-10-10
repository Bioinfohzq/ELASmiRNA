from autogluon.tabular import TabularDataset,TabularPredictor
from autogluon import tabular
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix as CM, precision_score as P, recall_score as R, f1_score as F1, accuracy_score as Acc
import math
from sklearn.model_selection import train_test_split

#读取特征
#导入miRNA提取好的正样本特征
miRNAkmer_P=pd.read_csv('/home/hzq/小论文data/miRNA/kmer_P.csv',index_col=0)
miRNADAC_P=pd.read_csv('/home/hzq/小论文data/miRNA/DAC_P.csv',index_col=0)
miRNA_P=pd.concat((miRNADAC_P,miRNAkmer_P),axis=1)
miRNA_P['label']=1      #给miRNA正样本打标签
miRNA_P
#导入miRNA提取好的负样本特征
miRNAkmer_N=pd.read_csv('/home/hzq/小论文data/miRNA/kmer_N.csv',index_col=0)
miRNADAC_N=pd.read_csv('/home/hzq/小论文data/miRNA/DAC_N.csv',index_col=0)
miRNA_N=pd.concat((miRNADAC_N,miRNAkmer_N),axis=1)
miRNA_N['label']=0      #给miRNA正样本打标签
miRNA_N
miRNA_traindata=pd.concat((miRNA_P,miRNA_N),axis=0) #所有的训练数据
#读取特征
#导入demiRNA提取好的正样本特征
demiRNAkmer_P=pd.read_csv('/home/hzq/小论文data/miRNA/dekmer_P.csv',index_col=0)
demiRNADAC_P=pd.read_csv('/home/hzq/小论文data/miRNA/deDAC_P.csv',index_col=0)
#print(miRNAkmer_P)
demiRNA_P=pd.concat((demiRNADAC_P,demiRNAkmer_P),axis=1)
demiRNA_P['label']=1      #给miRNA正样本打标签
#miRNA_P
#导入demiRNA提取好的负样本特征
demiRNAkmer_N=pd.read_csv('/home/hzq/小论文data/miRNA/dekmer_N.csv',index_col=0)
demiRNADAC_N=pd.read_csv('/home/hzq/小论文data/miRNA/deDAC_N.csv',index_col=0)
demiRNA_N=pd.concat((demiRNADAC_N,demiRNAkmer_N),axis=1)
demiRNA_N['label']=0      #给miRNA正样本打标签
miRNA_testdata=pd.concat((demiRNA_P,demiRNA_N),axis=0)  #所有的测试数据
miRNA=pd.concat((miRNA_traindata,miRNA_testdata),axis=0)

#划分数据
X=miRNA.iloc[:,:-1]  #所有特征
y=miRNA.iloc[:,-1]   #所有标签

#列索引进行保存
Qindex=miRNA.columns
feature_index=Qindex[0:-1]
#所有行索引进行保存
index_list=list(miRNA.index.values)
#对特征进行归一化
scaler_result = MinMaxScaler().fit_transform(X)
Qdata = pd.DataFrame(data=scaler_result,index=index_list,columns=feature_index)

#划分
trainX=Qdata.iloc[:752]   #训练集特征
trainy=y.iloc[:752]    #训练集标签
testX=Qdata.iloc[752:]    #独立测试集特征
testy=y.iloc[752:]     #独立测试集标签

Xtrain,Xtest,Ytrain,Ytest = train_test_split(trainX,trainy,test_size=0.3)
A = pd.concat((Xtrain,Xtest,testX),ignore_index=True)
B = pd.concat((Ytrain,Ytest,testy),ignore_index=True)

def feature_solve(FeatureSelect):
    train_feature = FeatureSelect.iloc[:526]    #训练集的筛选后的所有特征
    train_label = B.iloc[:526]             #训练集的所有标签
    train_data = pd.concat([pd.DataFrame(train_feature),pd.DataFrame(train_label)],axis=1)#用筛选后的特征和标签重新组合成新的特征矩阵，用于训练

    val_feature = FeatureSelect.iloc[526:752]     #独立测试集筛选后的所有特征
    val_label = B.iloc[526:752]           #独立测试集的所有标签

    test_feature = FeatureSelect.iloc[752:]
    test_label = B.iloc[752:]
    return train_feature,train_label,train_data,val_feature,val_label,test_feature,test_label
def myROC(y,y_score):   #y_score 是置信度分数的正样本概率的一列
    fpr, tpr, thersholds = roc_curve(y,y_score)
    AUC_score = auc(fpr, tpr)
    precisions, recalls, thresholds = precision_recall_curve(y, y_score)
    prc_score = average_precision_score(y, y_score)
    return fpr,tpr,AUC_score,precisions,recalls,prc_score

def myFeval(label, pred): # 定义你自己的评价标准
    #验证集指标
    cm=CM(label,pred,labels=[1,0])
    ACC=Acc(label,pred)
    #Acc=(cm[1,1]+cm[0,0])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    Pre=P(label,pred,labels=[1,0])
    Sen=R(label,pred,labels=[1,0])   #recall,TPR
    F_score=F1(label,pred,labels=[1,0])
    #PPV=cm[0,0]/cm[:,0].sum()  #这里的0,1是索引值   PPV=Pre
    FPR=cm[1,0]/cm[1,:].sum()
    NPV=cm[1,1]/cm[:,1].sum()
    Spe=cm[1,1]/cm[1,:].sum()
    MCC=(cm[0,0]*cm[1,1]-cm[1,0]*cm[0,1])/math.sqrt((cm[0,0]+cm[1,0])*(cm[0,0]+cm[0,1])*(cm[1,1]+cm[1,0])*(cm[1,1]+cm[0,1]))
    return print("Sen={0} Spe={1} Acc={2} Pre={3} F1={4} FPR={5} NPV={6} MCC={7}".format(Sen, Spe, ACC, Pre, F_score, FPR, NPV, MCC))

#训练
#训练
# Autogluon   10fold交叉验证结果  以及训练好的模型对独立测试集的预测结果

X_wrapper1 = RFE(SVC(kernel="linear"), n_features_to_select=300, step=50).fit_transform(A,B)
X_wrapper1=pd.DataFrame(X_wrapper1)
train_feature,train_label,train_data,val_feature,val_label,test_feature,test_label=feature_solve(X_wrapper1)
train = sklearn.utils.shuffle(train_data)
train = tabular.TabularDataset(train)    #用于autogluon模型训练数据需要带有标签的特征矩阵  这里等同于把train_data加载进来

save_path = '/home/hzq/anaconda3/jupyter.list/AutogluonModels/miRNA/agl116'
predictor = TabularPredictor(label='label',path = save_path,eval_metric='roc_auc').fit(train_data=train
                                                                                        ,presets='best_quality'
                                                                                        ,num_bag_folds=10
                                                                                       )
#验证结果
val_data = tabular.TabularDataset(val_feature)
predictor=TabularPredictor.load("/home/hzq/anaconda3/jupyter.list/AutogluonModels/miRNA/agl116")
val_pred = predictor.predict(val_data)
val_proba = predictor.predict_proba(val_data)
#该接口可以直接打印5个最终模型指标
perf = predictor.evaluate_predictions(y_true=val_label, y_pred=val_proba, auxiliary_metrics=True)
#测试结果
test_data = tabular.TabularDataset(test_feature)
predictor=TabularPredictor.load("/home/hzq/anaconda3/jupyter.list/AutogluonModels/miRNA/agl116")
test_pred = predictor.predict(test_data)
test_proba = predictor.predict_proba(test_data)
#print("Predictions:  \n", y_pred)
perf = predictor.evaluate_predictions(y_true=test_label, y_pred=test_proba, auxiliary_metrics=True)
myFeval(val_label, val_pred)   #交叉验证结果
myFeval(test_label, test_pred)   #训练好的模型对独立测试集的预测结果
val_score = val_proba.iloc[:,1]
test_score = test_proba.iloc[:,1]
aglfpr,agltpr,aglroc_auc,aglprecisions,aglrecalls,aglprc_score = myROC(val_label,val_score)
de_aglfpr,de_agltpr,de_aglroc_auc,de_aglprecisions,de_aglrecalls,de_aglprc_score = myROC(test_label,test_score)



