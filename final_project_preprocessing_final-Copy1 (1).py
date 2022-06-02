

def Automated_ML():
    import pandas as pd 
    import numpy as np
    data=pd.read_csv('customers.csv')
    #print(data.columns)
    output_column=input("enter output column name")
    for i in range(0,len(data.columns.values.tolist())):
        if output_column==data.columns.values.tolist()[i]:
            print(output_column)#1
    #column data type check and convert it into numerical form if not
    #check row size and apply restriction
    def data_conversion_and_data_transformation(df):
        #df=pd.read_csv(data)#reading the 
        df
        df=pd.DataFrame(data)
        #print(df)
        for i in range(0,df.shape[1]):
            if df[df.columns[i]].dtypes=='O':
                if (len(df[df.columns[i]].unique().tolist()))<=5:
                    k=0
                    t1=df[df.columns[i]].unique().tolist()
                    l=[k+j for j in range(0,len(df[data.columns[i]].unique().tolist()))]
                    #print(t1,l)
                    df[df.columns[i]].replace(t1,l,inplace=True)
        #print(df)
        def correlation(df,output_column):
            #print(df)
            corre=df.corr()[output_column]
            #print(corre)
            corr_column=corre.index.values
            #print(corr_column)
            corre=corre.tolist()
            #print(len(corre))
            dropped_column=[]
            for i in range(0,len(corre)):
                if ((corre[i]<0.1) & (corre[i]==1))| (corre[i]<-0.1):
                    dropped_column.append(corr_column[i])
                #print(dropped_column)
            #print(df)
            df.drop(axis=1,columns=dropped_column,inplace=True)
            #print(dropped_column)
        correlation(df,output_column)
        #print(df)
        #print(df1.columns.values)
        #make custom button for user to take some info to drop some columns like IDs,roll_no,index,or something that only 
        #shows serial no. 
        remove_columns=[]
        no_of_column_to_be_removed=int(input("enter no. of columns to be removed\n"))
        for i in range(0,no_of_column_to_be_removed):
            i=str(input("enter exact column name to be dropped\n"))
            remove_columns.append(i)
        df.drop(columns=remove_columns,inplace=True)
        #print(df)
        return df
    data=data_conversion_and_data_transformation(data)
    def null_value_count(df):
        #Data retrieval and conversion into dataframe
        #data conversion not done
        null_count=0
        rows_to_be_dropped=[]
        columns_to_be_dropped=[]
        temp=0
        for i in range(0,df.shape[0]):
            for j in range(0,df.shape[1]):
                if df.isnull().values[i][j]==True:
                    null_count=null_count+1
                    temp=temp+1
            if temp>=(0.7*df.shape[1]):#criteria for dropping rows 
                rows_to_be_dropped.append(i)
        for i in range(0,df.shape[1]):
            for j in range(0,df.shape[0]):
                if np.isnan(df.loc[1][df.columns[0]])==True:
                    temp=temp+1
            if temp>=(0.4*df.shape[0]):#criteria for dropping columns
                columns_to_be_dropped.append(df.columns[i])
        #print(null_count)
        def null_column_dropping_and_dataset_category_and_null_value_filling(df,rows_to_be_dropped,columns_to_be_dropped):
            df.dropna(axis=0,subset=rows_to_be_dropped,inplace=True)
            df.dropna(axis=1,subset=columns_to_be_dropped,inplace=True)
        #here we have to check whether the columns values or categorical or not
            def null_value_filling(df):
                def outlier_detection_and_null_value_filling(df):
                    outlier=[]
                    for i in range(0,df.shape[1]):
                        sort_data=np.sort(df[df.columns[i]])
                        Q1 = np.percentile(df, 25, interpolation = 'midpoint')  
                        Q3 = np.percentile(df, 75, interpolation = 'midpoint') 
                        IQR=Q3-Q1
                        low_lim = Q1 - 1.5 * IQR
                        up_lim = Q3 + 1.5 * IQR
                        column_with_null_and_outlier=[]
                        for y in df[df.columns[i]].values:
                            if (y> up_lim) or (y<low_lim):
                                outlier.append(y)
                                column_with_null_and_outlier.append(df.columns[i])
                        if len(outlier)!=0:
                            for i in range(0,data.shape[0]):
                                for j in range(0,data.shape[1]):
                                    if df.isnull().values[i][j]:
                                #print(data.iloc[i,j])
                                        column_with_null_and_outlier.append(df.columns[j])
                            for k in range(0,len(column_with_null_and_outlier)):
                                df[column_with_null_and_outlier[k]]=df[column_with_null_and_outlier[k]].fillna(df[column_with_null_and_outlier[k]].median())
                outlier_detection_and_null_value_filling(df)
                def categorical_or_numerical(df):
                    column_with_null=[]
                    if df[output_column].unique().shape[0]>5:
                        for i in range(0,df.shape[0]):
                            for j in range(0,df.shape[1]):
                                if data.isnull().values[i][j]:
                                #print(data.iloc[i,j])
                                    column_with_null.append(df.columns[j])
                        for k in range(0,len(column_with_null)):
                                df[column_with_null[k]]=df[column_with_null[k]].fillna(df[column_with_null[k]].mean())
                    else:
                        for i in range(0,df.shape[0]):
                            for j in range(0,df.shape[1]):
                                if df.isnull().values[i][j]:
                                #pr(data.iloc[i,j])
                                    column_with_null.append(df.columns[j])
                        for k in range(0,len(column_with_null)):
                                df[column_with_null[k]]=df[column_with_null[k]].fillna(df[column_with_null[k]].mode())
                categorical_or_numerical(df)
            null_value_filling(df)
        null_column_dropping_and_dataset_category_and_null_value_filling(df,rows_to_be_dropped,columns_to_be_dropped)
    null_value_count(data)
    def dataset_splitting(df):
        from sklearn.model_selection import train_test_split
        x=df.drop(axis=1,columns=output_column)
        y=df[output_column]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.33,shuffle=True) 
        return x_train,x_test,y_train,y_test,x_val,y_val
    def machine_learning_model(df,output_column):
        #print('hello')

        from math import isclose
        x_train,x_test,y_train,y_test,x_val,y_val=dataset_splitting(data)
        from sklearn.model_selection import train_test_split
        x=df.drop(axis=1,columns=output_column)
        y=df[output_column]
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.33,shuffle=True) 
        before_standarization(df,x_train,y_train,x_test,y_test)
        df=standardization(df)
        print(df)
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.33,shuffle=True)
        
        before_standarization(df,x_train,y_train,x_test,y_test)
        #return x_train,y_train,x_test,y_test,x_val,y_val 
        #before_standarization(df,x_train,y_train,x_test,y_test)
        #dataset_splitting(df)
        #print(x_train)
    #machine_learning_model(data,output_column)
    def before_standarization(df,x_train,y_train,x_test,y_test):
        if len(df[output_column].unique())<5:
            def model_for_categorical_data():  
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.naive_bayes import MultinomialNB
                from sklearn import svc
                from sklearn import metrics
                from sklearn.metrics import mean_absolute_error

                def Knn():
                    knn=KNeighborsClassifier()
                    knn.fit(x_train,y_train)
                    y_pred_knn=knn.predict(x_test)
                    y_train_pred = knn.predict(x_train)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mae_train.append(mean_absolute_error(y_train, y_train_pred))
                    test_mae=mae_test.append(mean_absolute_error(y_test, y_pred_knn))
                    print(y_test,y_pred_knn)
                    accuracy_of_knn=metrics.accuracy_score(y_test,y_pred_knn)
                    print("accuracy:\n")
                knn()
                def LogReg():
                    logistic=LogisticRegression()
                    logistic.fit(x_train,y_train)
                    y_pred_logistic=logistic.predict(x_test)
                    y_train_pred = logistic.predict(x_train)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mae_train.append(mean_absolute_error(y_train, y_train_pred))
                    test_mae=mae_test.append(mean_absolute_error(y_test, y_pred_logistic))
                    print(y_test,y_pred_logistic)
                    accuracy_of_logistic=metrics.accuracy_score(y_test,y_pred_knn)
                    print("accuracy:\n")
                     
                LogReg()
                def DeciTree():
                    dectree=DecisionTreeClassifier()
                    dectree.fit(x_train,y_train)
                    y_pred_dectree=dectree.predict(x_test)
                    y_train_pred = dectree.predict(x_train)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mae_train.append(mean_absolute_error(y_train, y_train_pred))
                    test_mae=mae_test.append(mean_absolute_error(y_test, y_pred_dectree))
                    print(y_test,y_pred_dectree)
                    accuracy_of_dectree=metrics.accuracy_score(y_test,y_pred_dectree)
                    print("accuracy:\t")
                    
                DeciTree()
                def RanFor():
                    random_forest=RandomForestClassifier()
                    random_forest.fit(x_train,y_train)
                    y_pred_random_forest=random_forest.predict(x_test)
                    y_train_pred = random_forest.predict(x_train)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mae_train.append(mean_absolute_error(y_train, y_train_pred))
                    test_mae=mae_test.append(mean_absolute_error(y_test, y_pred_random_forest))
                    print(y_test,y_pred_random_forest)
                    accuracy_of_random_forest=metrics.accuracy_score(y_test,y_pred_random_forest)
                    print("accuracy:\t")
                     
                RanFor()
                def SVC():
                    support_vc=svc()
                    support_vc.fit(x_train,y_train)
                    y_pred_support_vc=support_vc.predict(x_test)
                    y_train_pred = support_vc.predict(x_train)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mae_train.append(mean_absolute_error(y_train, y_train_pred))
                    test_mae=mae_test.append(mean_absolute_error(y_test, y_pred_support_vc))
                    print(y_test,y_pred_support_vc)
                    accuracy_of_svc=metrics.accuracy_score(y_test,y_pred_support_vc)
                    print("accuracy:\t")
                    
                    
                SVC()
            model_for_categorical_data()
        else:
            diff={}
            def model_for_numerical_data():
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.linear_model import LinearRegression
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.svm import SVR
                from sklearn import metrics
                from sklearn.metrics import mean_absolute_error
                import matplotlib.pyplot as plt
                #def standardization_and_normalization(df):

                def Knr():
                    knr=KNeighborsRegressor(n_neighbors=3)
                    knr.fit(x_train,y_train)
                    y_pred_knr=knr.predict(x_test)
                    y_train_pred =knr.predict(x_train)
                    print(x_test.shape,x_train.shape,y_test.shape,y_train.shape,y_pred_knr.shape,y_train_pred.shape)
                    mae_train=[]
                    mae_test=[]
                    train_mae=mean_absolute_error(y_train, y_train_pred)
                    test_mae=mean_absolute_error(y_test, y_pred_knr)
                    diff['Knr']=test_mae-train_mae
                    print(diff['Knr'])
                    plt.scatter(np.array(y_pred_knr),y_test,c='blue')
                    plt.scatter(y_train,y_train_pred,c='pink')
                    plt.show()
                Knr()
                def LinReg():
                    linear=LinearRegression()
                    linear.fit(x_train,y_train)
                    y_pred_linear=linear.predict(x_test)
                    y_train_pred = linear.predict(x_train)
                    train_mae=mean_absolute_error(y_train, y_train_pred)
                    test_mae=mean_absolute_error(y_test, y_pred_linear)
                    diff['LinReg']=test_mae-train_mae
                    print(diff['LinReg'])
                    plt.scatter(np.array(y_pred_linear),y_test,c='blue')
                    plt.scatter(y_train,y_train_pred,c='pink')
                    plt.show()
                LinReg()
                def DeciTreeReg():
                    dectree=DecisionTreeRegressor()
                    dectree.fit(x_train,y_train)
                    y_pred_dectree=dectree.predict(x_test)
                    y_train_pred = dectree.predict(x_train)
                    train_mae=mean_absolute_error(y_train, y_train_pred)
                    test_mae=mean_absolute_error(y_test, y_pred_dectree)
                    diff['DeciTreeReg']=test_mae-train_mae
                    print(diff['DeciTreeReg'])
                    plt.scatter(np.array(y_pred_dectree),y_test,c='blue')
                    plt.scatter(y_train,y_train_pred,c='pink')
                    plt.show()

                DeciTreeReg()
                def RanForReg():
                    random_forest=RandomForestRegressor()
                    random_forest.fit(x_train,y_train)
                    y_pred_random_forest=random_forest.predict(x_test)
                    y_train_pred = random_forest.predict(x_train)
                    train_mae=mean_absolute_error(y_train, y_train_pred)
                    test_mae=mean_absolute_error(y_test, y_pred_random_forest)
                    diff['RanForReg']=test_mae-train_mae
                    print(diff['RanForReg'])
                    plt.scatter(np.array(y_pred_random_forest),y_test,c='blue')
                    plt.scatter(y_train,y_train_pred,c='pink')
                    plt.show()
                RanForReg()
                def SVReg():
                    support_vr=SVR()
                    support_vr.fit(x_train,y_train)
                    y_pred_support_vr=support_vr.predict(x_test)
                    y_train_pred = support_vr.predict(x_train)
                    train_mae=mean_absolute_error(y_train, y_train_pred)
                    test_mae=mean_absolute_error(y_test, y_pred_support_vr)
                    diff['SVReg']=test_mae-train_mae
                    print(diff['SVReg'])
                    plt.scatter(np.array(y_pred_support_vr),y_test,c='blue')
                    plt.scatter(y_train,y_train_pred,c='pink')
                    plt.show()


                SVReg()
                return df
            model_for_numerical_data()
    #before_standarization(data,x_train,y_train,x_test,y_test)
    def standardization(df):
            def min_max_scaling():
                df_min_max_scaled=df.copy()
                df_columns=df_min_max_scaled.columns.values.tolist()
                print(df_columns)
                for i in range(0,len(df_columns)):
                    print(i)
                    df_min_max_scaled[df_columns[i]]=(df_min_max_scaled[df_columns[i]] - df_min_max_scaled[df_columns[i]].min()) / (df_min_max_scaled[df_columns[i]].max() - df_min_max_scaled[df_columns[i]].min())
                return df_min_max_scaled
            df1=min_max_scaling()
            return df1
    machine_learning_model(data,output_column)
    
        





# In[14]:


Automated_ML()


# In[ ]:


col=['jnfvdjf','jnvdfjkn']
for i in col:
    print(i)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.scatter([1,2],[3,4],c='blue')
plt.show()


# In[ ]:




