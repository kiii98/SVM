import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss

def _preprocess():
    df_train = pd.read_csv('bank/bank-full.csv',sep = ';')
    df_test = pd.read_csv('bank/bank.csv',sep = ';')
    category_cols = {'job','marital','education','default','housing','loan','contact','poutcome'}
    numerical_cols = {'age','balance','day','duration','campaign','pdays','previous','month'}

    label_train = df_train.y
    label_train.replace('yes', 1,inplace = True)
    label_train.replace('no', 0,inplace = True)

    label_test = df_test.y
    label_test.replace('yes', 1,inplace = True)
    label_test.replace('no', 0,inplace = True)

    df_train.drop(['y'],axis=1,inplace=True)
    df_test.drop(['y'], axis=1, inplace=True)

    ntrain = df_train.shape[0]

    merge_data = pd.concat((df_train, df_test)).reset_index(drop=True)

    #######################Age###########################
    fig, ax = plt.subplots()
    ax.hist(merge_data['age'])
    plt.title('Distribution of Age')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.show()
    bins_of_Age = [15, 30, 50, 60, 100]
    merge_data['age'] = pd.cut(merge_data['age'], bins_of_Age,labels=[1,2,3,4])

    #####################balance###################################
    fig, ax = plt.subplots()
    ax.hist(merge_data['balance'],range=[-8000,30000])
    plt.title('Distribution of balance')
    plt.xlabel('balance')
    plt.ylabel('Count')
    plt.show()
    bins_of_balance = [-9000, 0 ,5000, 10000, 20000,110000]
    merge_data['balance'] = pd.cut(merge_data['balance'], bins_of_balance, labels=[1, 2, 3, 4, 5])

    #####################day and month ###################################
    merge_data['month'].replace('jan', 365,inplace = True)
    merge_data['month'].replace('feb', 334,inplace = True)
    merge_data['month'].replace('mar', 305,inplace = True)
    merge_data['month'].replace('apr', 274,inplace = True)
    merge_data['month'].replace('may', 244,inplace = True)
    merge_data['month'].replace('jun', 214,inplace = True)
    merge_data['month'].replace('jul', 184,inplace = True)
    merge_data['month'].replace('aug', 153,inplace = True)
    merge_data['month'].replace('sep', 122,inplace = True)
    merge_data['month'].replace('oct', 92,inplace = True)
    merge_data['month'].replace('nov', 61,inplace = True)
    merge_data['month'].replace('dec', 31,inplace = True)

    merge_data['days_cnt'] = merge_data['month'] - merge_data['day']
    merge_data.drop(['month'], axis=1, inplace=True)
    merge_data.drop(['day'], axis=1, inplace=True)

    fig, ax = plt.subplots()
    ax.hist(merge_data['days_cnt'])
    plt.title('Distribution of days_cnt')
    plt.xlabel('days_cnt')
    plt.ylabel('count')
    plt.show()
    bins_of_day = [-1, 90, 180, 365]
    merge_data['days_cnt'] = pd.cut(merge_data['days_cnt'], bins_of_day, labels=[1, 2, 3])

    #####################duration###################################
    merge_data['duration'] = merge_data['duration'] / 60.0
    fig, ax = plt.subplots()
    ax.hist(merge_data['duration'],range = [0,250])
    plt.title('Distribution of duration')
    plt.xlabel('duration')
    plt.ylabel('Count')
    plt.show()
    bins_of_duration = [-1, 25 ,50, 90]
    merge_data['duration'] = pd.cut(merge_data['duration'], bins_of_duration, labels=[1, 2, 3])

    #####################campaign###################################
    fig, ax = plt.subplots()
    ax.hist(merge_data['campaign'],range = [0,40])
    plt.title('Distribution of campaign')
    plt.xlabel('campaign')
    plt.ylabel('Count')
    plt.show()
    bins_of_campaign = [0, 10, 20, 65]
    merge_data['campaign'] = pd.cut(merge_data['campaign'], bins_of_campaign, labels=[1, 2, 3])

    #####################pdays###################################
    fig, ax = plt.subplots()
    ax.hist(merge_data['pdays'])
    plt.title('Distribution of pdays')
    plt.xlabel('pdays')
    plt.ylabel('Count')
    plt.show()
    bins_of_pdays = [-2, 0, 100, 200,400,900]
    merge_data['pdays'] = pd.cut(merge_data['pdays'], bins_of_pdays, labels=[1, 2, 3,4,5])

    #####################previous###################################
    fig, ax = plt.subplots()
    ax.hist(merge_data['previous'],range = [0,50])
    plt.title('Distribution of previous')
    plt.xlabel('previous')
    plt.ylabel('Count')
    plt.show()
    bins_of_previous = [-1, 5, 10, 20, 300]
    merge_data['previous'] = pd.cut(merge_data['previous'], bins_of_previous, labels=[1, 2, 3, 4])

    for col in category_cols:
        if 'unknown' in  merge_data[col]:
            ratio = merge_data[col].value_counts()['unknown'] / len(merge_data[col])
            if ratio < 0.1:
                merge_data[col].replace('unknown',merge_data[col].mode()[0])
        lbl = LabelEncoder()
        lbl.fit(list(merge_data[col].values))
        merge_data[col] = lbl.transform(list(merge_data[col].values))
    print(merge_data.isnull().sum())
    train = merge_data[:ntrain]
    test = merge_data[ntrain:]

    return train, label_train, test, label_test


def main():
    train, label_train, test, label_test = _preprocess()
    svm = SVC(C = 100,kernel='rbf', class_weight='balanced',random_state = 1010)
    svm.fit(train, label_train)
    label_predict = svm.predict(test)
    print("log_loss = ",log_loss(label_test,label_predict))

if __name__ == '__main__':
    main()