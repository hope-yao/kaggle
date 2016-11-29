
import numpy as np
import scipy
from scipy.io import loadmat #be used to get data from .mat document
from scipy import signal #scipy: scientific python, be used to scientific computation and signal is used to procsess signal

import pywt #python wavelate transformation
from os import walk #walk: be used to read file name

import sklearn
from sklearn.decomposition import FastICA

import matplotlib #be used to plot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plt_allch(sig,c):
    num_ch = sig.shape[1]
    fig = plt.figure(figsize=(30,30))
    for i in range(num_ch):
        plt.subplot(num_ch, 1, i+1)
        plt.plot(range(sig.shape[0]),sig[:,i],c)
    return fig

'''read feature data'''
def read_fea_train(person,data_safe):
    feapath = ['/home/hope-yao/Documents/kaggle/projects/featurefiles/train_',str(person)]
    destpath = ''.join(feapath)
    f = []
    for (dirpath, dirnames, filenames) in walk(destpath):
        for fn in filenames:
            if fn[-3:]=='npy':
                f.extend([fn])
        break
    file_path = []
    for x in f:
        file_path.extend([dirpath+'/'+x])

    ttnf = len(file_path)
    print('total number of training feature files found:',ttnf)

    n = ttnf #ttnf
    fea_sig = []
    fea_label = []
    for i,fp in enumerate(file_path[0:n]):
        idx_tmp = int(fp[fp.rfind('/')+3:len(fp)-6])
        if fp[-5]:
            if person ==1:
                flg_safe = data_safe[1152 + idx_tmp]
            elif person ==2:
                flg_safe = data_safe[3498 + idx_tmp]
            elif person ==3:
                flg_safe = data_safe[5892 + idx_tmp]
            else:
                print('error')
                break
        else:
            flg_safe = data_safe[idx_tmp * 6]

        # skip contaminated data
        if flg_safe==0:
            continue

        fea_data = np.load(fp)
        tmp = fea_data # inside structure
        if  tmp.shape==(6,16):  # check signal size
            if (np.count_nonzero(tmp)):
                if 0:
                    tmp_onehour = []
                    for i in range(tmp.shape[0]):
                        tmp_onehour = tmp_onehour + tmp[i,:].tolist()
                    fea_sig.append(tmp_onehour) #nf ,raw_data_length,num_ch
                    fea_label.append(int(fp[-5]))
                else:
                    for i in range(tmp.shape[0]):
                        fea_sig.append(tmp[i,:]) #nf ,raw_data_length,num_ch
                        fea_label.append(int(fp[-5]))
        else:
#             print('file with missing clip found!',fp)
            continue
    fea_sig = np.asarray(fea_sig)
#     print(fea_sig.shape)
#     print(len(fea_label))
    return fea_sig,fea_label

def read_fea_test(person):
    feapath = ['/home/hope-yao/Documents/kaggle/projects/featurefiles/test_',str(person),'_new']
    destpath = ''.join(feapath)
    f = []
    for (dirpath, dirnames, filenames) in walk(destpath):
        for fn in filenames:
            if fn[-3:]=='npy':
                f.extend([fn])
        break
    file_path = []
    for x in f:
        file_path.extend([dirpath+'/'+x])

    ttnf = len(file_path)
    print('total number of testing feature files found:',ttnf)

    n = ttnf #ttnf
    fea_sig = []
    for i,fp in enumerate(file_path[0:n]):
        fea_data = np.load(fp)
        tmp = fea_data # inside structure
        if  tmp.shape==(6,16):  # check signal size
            if (np.count_nonzero(tmp)):
                if 0:
                    tmp_onehour = []
                    for i in range(tmp.shape[0]):
                        tmp_onehour = tmp_onehour + tmp[i,:].tolist()
                    fea_sig.append(tmp_onehour) #nf ,raw_data_length,num_ch
                else:
                    for i in range(tmp.shape[0]):
                        fea_sig.append(tmp[i,:]) #nf ,raw_data_length,num_ch
        else:
#             print('file with missing clip found!',fp)
            continue
    fea_sig = np.asarray(fea_sig)
#     print(fea_sig.shape)
    return fea_sig


'''train_and_test_data_labels_safe'''
def check_safety():
    import csv
    data_name = []
    data_safe = []
    data_class = []
    with open('./datafiles/train_and_test_data_labels_safe.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in csvreader:
            data_class = data_class + [int(row[0][-3])]
            data_safe = data_safe + [int(row[0][-1])]
            data_name = data_name + [row[0][0:-8]]
    return data_class,data_safe,data_name


'''data balance and training/testing split'''


def over_sample(fea_sig, fea_label, split):
    from imblearn.over_sampling import SMOTE
    from random import sample

    # Apply regular SMOTE
    sm = SMOTE(kind='regular')
    data_balance, label_balance = sm.fit_sample(fea_sig, fea_label)
    l = len(data_balance)  # length of data
    f = int(split * l)  # split for testing
    indices = sample(range(l), f)

    data_test = data_balance[indices]
    label_test = label_balance[indices]

    data_train = np.delete(data_balance, indices, 0)
    label_train = np.delete(label_balance, indices, 0)

    return data_train, label_train, data_test, label_test


'''Random Forest'''
def train_rf(data_train,label_train):
    #from sklearn.model_selection import GridSearchCV
    from sklearn.grid_search import RandomizedSearchCV
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from time import time
    from scipy.stats import randint as sp_randint

    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    clf = RandomForestClassifier(n_estimators=200,n_jobs=-1)

    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()


    random_search.fit(data_train, label_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    #report(random_search.cv_results_)
    return random_search


def draw_roc(random_search, data_test, label_test):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, auc

    y_pred = random_search.predict(data_test)
    RFprobaR = random_search.predict_proba(data_test)
    roc_score = roc_auc_score(np.array(label_test), np.array(RFprobaR[:, 1]))
    print ("ROC score    : {:.4f}.".format(roc_score))

    fpr, tpr, _ = roc_curve(label_test, RFprobaR[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating curve- Random Forest')
    plt.legend(loc="lower right")
    plt.show(True)

    return roc_score, y_pred, RFprobaR

def plt_cfm(y_pred,label_test):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Compute confusion matrix for a model
    cm=confusion_matrix(y_pred,label_test)

    # view with a heatmap
    sns.heatmap(cm, annot=True, cmap='RdBu', xticklabels=['no', 'yes'], yticklabels=['no', 'yes'], linewidth=.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix for:\n{}'.format(" Random Forest "))
    return plt

all_results = []
for person in range(1,4):
    # check if data has been contaminated
    data_class,data_safe,data_name = check_safety()
    # read feature training data
    fea_sig,fea_label = read_fea_train(person,data_safe)
    # balance data and split it
    split = 0.2
    data_train,label_train,data_test,label_test = over_sample(fea_sig, fea_label, split)
    # train random forest
    random_search = train_rf(data_train,label_train)
    # predict testing data and plot roc curve
    roc_score,y_pred,RFprobaR = draw_roc(random_search,data_test,label_test)
    # plot confusion matrix
    plt_cfm(y_pred,label_test)

    # read feature testing data to do prediction
    fea_sig = read_fea_test(person)
    RFprobaR=random_search.predict_proba(fea_sig)
    all_results = all_results + [RFprobaR[:,1]]
    print('****************************************patient index {:d} prediction done!'.format(person))

import pandas as pd
y=pd.DataFrame(np.concatenate((all_results[0], all_results[1], all_results[2]), axis=0))
y.to_csv("result.csv")









