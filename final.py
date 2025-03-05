from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
import numpy as np
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense# Neural network
from sklearn.naive_bayes import GaussianNB

main = tkinter.Tk()
main.title("Student Acceptances")
main.geometry("1300x1200")

def upload():
    global filename
    global data
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def importdata():
    global filename
    global dataset
    dataset = pd.read_csv(filename)
    text.insert(END,"Data Information:\n"+str(dataset.head())+"\n")
    text.insert(END,"Columns Information:\n"+str(dataset.columns)+"\n")
    accept_list = dataset[dataset['status'] == 'accept']
    reject_list = dataset[dataset['status'] == 'reject']
    plt.bar('accept',len(accept_list))
    plt.bar('reject',len(reject_list))
    plt.title('Accept and Reject for Universities')
    plt.show()

    print(dataset['status'].value_counts())
    print('===================================')
    print('Percentage of Accept and Reject Values in the dataset')
    

def preprocess():
    global X,y
    global dataset
    university_list = dataset['university_name'].unique().tolist()
    text.insert(END,"Universities Information: \n"+str('universities') + '\n')
    
    dataset.groupby(by=["university_name"]).mean()["ranking"].sort_values()
    dataset.groupby(['university_name','ranking','status'])['status'].count().unstack().plot(title = 'University_name vs Admit/Reject',fontsize = 30,figsize=(50,15),kind='bar', legend=False, color=['g', 'r'])

    dataset.groupby(['university_name','status'])['test_score_toefl'].mean().unstack().plot(legend=True,ylim = [85,120],title = 'Accept and Reject of universities on the basis on mean TOEFL Score',fontsize = 30,figsize=(50,15),kind='bar', color=['g','r'])
    dataset.university_name.value_counts()
    target_universities=dataset.university_name.unique().tolist()

    resampled_dfs=[]
    resampled_df = pd.DataFrame()
    for each in target_universities:
        if dataset[(dataset.university_name==each )].shape[0]> 600:
            
            resampled_dfs.append(resample(dataset[(dataset.university_name==each )&(dataset.status=='accept')],replace=True,n_samples=300,random_state=123))
            resampled_dfs.append(resample(dataset[(dataset.university_name==each) &(dataset.status=='reject')],replace=True,n_samples=300,random_state=123))
            
        elif dataset[(dataset.university_name==each )].shape[0] < 200:
            resampled_dfs.append(resample(dataset[(dataset.university_name==each )&(dataset.status=='accept')],replace=True,n_samples=125,random_state=123))
            resampled_dfs.append(resample(dataset[(dataset.university_name==each) &(dataset.status=='reject')],replace=True,n_samples=125,random_state=123))
        else:
            resampled_dfs.append(dataset[(dataset.university_name==each )&(dataset.status=='accept')])
            resampled_dfs.append(dataset[(dataset.university_name==each )&(dataset.status=='reject')])
            

    resampled_df = pd.concat( [ f for f in resampled_dfs ] )
        
    resampled_df.groupby(by='university_name')['status'].value_counts()

    dataset =resampled_df.copy()
    text.insert(END,"Preprocess Done: \n")
    plt.figure(figsize = (10, 6))
    ax =sns.boxplot(x='university_name', y='gre_score', data=dataset[dataset['status']=='accept'])
    plt.setp(ax.artists, alpha=1, linewidth=1, edgecolor="k")
    plt.title('GRE SCORE Box Plot - Accepts')
    plt.xticks(rotation=90)
    
    plt.figure(figsize = (10, 6))
    ax =sns.boxplot(x='university_name', y='gre_score_verbal', data=dataset[dataset['status']=='accept'])
    plt.setp(ax.artists, alpha=1, linewidth=1, edgecolor="k")
    plt.title('GRE SCORE VERBAL Box Plot - Accepts')
    plt.xticks(rotation=90)

    plt.figure(figsize = (10, 6))
    ax =sns.boxplot(x='university_name', y='gre_score_quant', data=dataset[dataset['status']=='accept'])
    plt.setp(ax.artists, alpha=1, linewidth=1, edgecolor="k")
    plt.title('GRE SCORE QUANT Box Plot - Accepts')
    plt.xticks(rotation=90)
    plt.show()

def ttmodel():
    global dataset
    global training, testing, numerical_features, categorical_features
    #train test split for modelling
    training, testing = train_test_split(dataset, test_size=0.25, random_state=5, stratify=dataset[['university_name', 'status']])
    testing.groupby(by=['university_name'])['status'].value_counts()
    numerical_data = training.select_dtypes(include = ['int64','float','uint8'])
    categorical_data = training.select_dtypes(include = ['object'])
    categorical_features = categorical_data.columns.values
    numerical_features = numerical_data.columns.values
    text.insert(END,"train Data Shape: "+str(training.shape)+"\n")
    text.insert(END,"test Data Shape: "+str(testing.shape)+"\n")


model_name=[]
model_train_acc=[]
model_test_accuracy=[]
model_train_f1=[]
model_test_f1=[]

def get_result(model, X_train, X_test, Y_train, Y_test):
    sc = StandardScaler() 
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test) 
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    prob_test=pd.DataFrame(model.predict_proba(X_test))
    prob_train=pd.DataFrame(model.predict_proba(X_train))
    test_f1_score = f1_score(Y_test, y_pred,pos_label='accept')
    train_f1_score = f1_score(Y_train, y_train_pred,pos_label='accept')
    train_accuracy=accuracy_score(Y_train, y_train_pred)
    test_accuracy=accuracy_score(Y_test, y_pred)
    test_cm = confusion_matrix(Y_test, y_pred,labels=['accept','reject'])
    train_cm = confusion_matrix(Y_train, y_train_pred,labels=['accept','reject'])
    model_name.append(model)
    model_train_acc.append(train_accuracy)
    model_test_accuracy.append(test_accuracy)
    model_test_f1.append(test_f1_score)
    model_train_f1.append(train_f1_score)
    return [train_cm,test_cm,train_accuracy,test_accuracy,train_f1_score, test_f1_score, prob_train,prob_test, y_pred,y_train_pred, model,sc]

def generate_cm_roc(model_results):
    test_fpr,test_tpr,test_thresholds = metrics.roc_curve(testing['status'], model_results[7][0],pos_label='accept')
    test_roc_auc = auc(test_fpr, test_tpr)
    train_fpr,train_tpr,train_thresholds = metrics.roc_curve(training['status'], model_results[6][0],pos_label='accept')
    train_roc_auc = auc(train_fpr, train_tpr)
    plt.plot(train_fpr, train_tpr, lw=2, alpha=0.5,
                 label='Train ROC (auc= %0.2f)' % (train_roc_auc))
    plt.plot(test_fpr, test_tpr, lw=2, alpha=0.5,
                 label='Test ROC (auc= %0.2f)' % (test_roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def svc():
    global svc_model_results
    svc_model=SVC(gamma='auto',probability=True)
    svc_model_results=get_result(svc_model,training[numerical_features],testing[numerical_features],training['status'],testing['status'])
    text.insert(END,'test_accuracy:'+str(svc_model_results[3])+"\n")
    text.insert(END,'train_accuracy:'+str(svc_model_results[2])+"\n")
    text.insert(END,'test_f1_score:'+str(svc_model_results[5])+"\n")
    text.insert(END,'train_f1_score:'+str(svc_model_results[4])+"\n")
    generate_cm_roc(svc_model_results)

def dt():
    global decision_tree_model_results
    decision_tree_model=DecisionTreeClassifier()
    decision_tree_model_results=get_result(decision_tree_model,training[numerical_features],testing[numerical_features],training['status'],testing['status'])
    text.insert(END,'test_accuracy:'+str(decision_tree_model_results[3])+"\n")
    text.insert(END,'train_accuracy:'+str(decision_tree_model_results[2])+"\n")
    text.insert(END,'test_f1_score:'+str(decision_tree_model_results[5])+"\n")
    text.insert(END,'train_f1_score:'+str(decision_tree_model_results[4])+"\n")
    decision_tree_model_results[10].get_params
    generate_cm_roc(decision_tree_model_results)


def rf():
    global random_forest_model_results
    random_forest_model=RandomForestClassifier(n_estimators=10)
    random_forest_model_results=get_result(random_forest_model,training[numerical_features],testing[numerical_features],training['status'],testing['status'])
    text.insert(END,'test_accuracy: s'+str(random_forest_model_results[3])+"\n")
    text.insert(END,'train_accuracy:'+str(random_forest_model_results[2])+"\n")
    text.insert(END,'test_f1_score:'+str(random_forest_model_results[5])+"\n")
    text.insert(END,'train_f1_score:'+str(random_forest_model_results[4])+"\n")
    generate_cm_roc(random_forest_model_results)

def nb():
    global navie_bayes_model_results
    navie_bayes_model = GaussianNB()
    navie_bayes_model_results=get_result(navie_bayes_model,training[numerical_features],testing[numerical_features],training['status'],testing['status'])
    text.insert(END,'test_accuracy:'+str(navie_bayes_model_results[3])+"\n")
    text.insert(END,'train_accuracy:'+str(navie_bayes_model_results[2])+"\n")
    text.insert(END,'test_f1_score:'+str(navie_bayes_model_results[5])+"\n")
    text.insert(END,'train_f1_score:'+str(navie_bayes_model_results[4])+"\n")
    generate_cm_roc(navie_bayes_model_results)


def ann():
    global ann_acc
    lb = preprocessing.LabelBinarizer()
    y_train = lb.fit_transform(training['status'])
    y_test = lb.fit_transform(testing['status'])

    model_ann = Sequential()
    model_ann.add(Dense(16, input_dim=8, activation='relu'))
    model_ann.add(Dense(12, activation='relu'))
    model_ann.add(Dense(1, activation='softmax'))
    model_ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model_ann.fit(training[numerical_features], y_train, epochs=200, batch_size=64)

    ann_acc = model_ann.evaluate(testing[numerical_features],y_test)
    text.insert(END,'test_accuracy:'+str(ann_acc[1])+"\n")

def graph():
    global navie_bayes_model_results,decision_tree_model_results,random_forest_model_results,svc_model_results,ann_acc
    
    height = [navie_bayes_model_results[3],decision_tree_model_results[3],random_forest_model_results[3],svc_model_results[3],ann_acc[1]]
    print(height,len(height))
    bars = ['NB', 'DT','RFC','SVC','ANN']
    print(bars,len(bars))
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   

font = ('times', 16, 'bold')
title = Label(main, text='Using Data Mining Techniques to Predict Student Performance to Support Decision Making in University Admission Systems')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

ip = Button(main, text="Data Import", command=importdata)
ip.place(x=700,y=200)
ip.config(font=font1)

pp = Button(main, text="Data Preprocessing", command=preprocess)
pp.place(x=700,y=250)
pp.config(font=font1)

tt = Button(main, text="Train and Test Model", command=ttmodel)
tt.place(x=700,y=300)
tt.config(font=font1)

sc = Button(main, text="Run Support Vector Machine", command=svc)
sc.place(x=700,y=350)
sc.config(font=font1)

rr = Button(main, text="Run RandomForest", command=rf)
rr.place(x=700,y=400)
rr.config(font=font1)

nn = Button(main, text="Run Navie Bayes", command=nb)
nn.place(x=700,y=450)
nn.config(font=font1)

nn = Button(main, text="Run Decision Tree", command=dt)
nn.place(x=700,y=500)
nn.config(font=font1)

aa = Button(main, text="Run ANN", command=ann)
aa.place(x=700,y=550)
aa.config(font=font1)

gph = Button(main, text="Accuracy Graph", command=graph)
gph.place(x=700,y=600)
gph.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='dim gray')
main.mainloop()

    



