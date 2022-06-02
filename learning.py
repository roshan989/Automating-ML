import numpy as np
import graph as gf
import logging as log
output_column = ''

def dataset_splitting(df):
    from sklearn.model_selection import train_test_split
    x = df.drop(axis=1, columns=output_column)
    y = df[output_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.33, shuffle=True)
    return x_train, x_test, y_train, y_test, x_val, y_val


def machine_learning_model(df):
    # print('hello')
    from math import isclose
    x_train, x_test, y_train, y_test, x_val, y_val = dataset_splitting(df)
    from sklearn.model_selection import train_test_split
    x = df.drop(axis=1, columns=output_column)
    y = df[output_column]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.33, shuffle=True)
    before_standarization(df, x_train, y_train, x_test, y_test)
    df = standardization(df)
    print(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.33, shuffle=True)

    before_standarization(df, x_train, y_train, x_test, y_test)
    # return x_train,y_train,x_test,y_test,x_val,y_val
    # before_standarization(df,x_train,y_train,x_test,y_test)
    # dataset_splitting(df)
    # print(x_train)
# machine_learning_model(data,output_column)


def before_standarization(df, x_train, y_train, x_test, y_test):
    
    if len(df[output_column].unique()) < 5:
        def model_for_categorical_data():
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.svm import SVC
            from sklearn import metrics
            from sklearn.metrics import mean_absolute_error

            def Knn():
                knn = KNeighborsClassifier()
                knn.fit(x_train, y_train)
                y_pred_knn = knn.predict(x_test)
                y_train_pred = knn.predict(x_train)
                mae_train = []
                mae_test = []
                train_mae = mae_train.append(
                    mean_absolute_error(y_train, y_train_pred))
                test_mae = mae_test.append(
                    mean_absolute_error(y_test, y_pred_knn))
                print(y_test, y_pred_knn)
                accuracy_of_knn = metrics.accuracy_score(y_test, y_pred_knn)
                print("accuracy:\n")
            knn()

            def LogReg():
                logistic = LogisticRegression()
                logistic.fit(x_train, y_train)
                y_pred_logistic = logistic.predict(x_test)
                y_train_pred = logistic.predict(x_train)
                mae_train = []
                mae_test = []
                train_mae = mae_train.append(
                    mean_absolute_error(y_train, y_train_pred))
                test_mae = mae_test.append(
                    mean_absolute_error(y_test, y_pred_logistic))
                print(y_test, y_pred_logistic)
                accuracy_of_logistic = metrics.accuracy_score(
                    y_test, y_pred_logistic)
                print("accuracy:\n")

            LogReg()

            def DeciTree():
                dectree = DecisionTreeClassifier()
                dectree.fit(x_train, y_train)
                y_pred_dectree = dectree.predict(x_test)
                y_train_pred = dectree.predict(x_train)
                mae_train = []
                mae_test = []
                train_mae = mae_train.append(
                    mean_absolute_error(y_train, y_train_pred))
                test_mae = mae_test.append(
                    mean_absolute_error(y_test, y_pred_dectree))
                print(y_test, y_pred_dectree)
                accuracy_of_dectree = metrics.accuracy_score(
                    y_test, y_pred_dectree)
                print("accuracy:\t")

            DeciTree()

            def RanFor():
                random_forest = RandomForestClassifier()
                random_forest.fit(x_train, y_train)
                y_pred_random_forest = random_forest.predict(x_test)
                y_train_pred = random_forest.predict(x_train)
                mae_train = []
                mae_test = []
                train_mae = mae_train.append(
                    mean_absolute_error(y_train, y_train_pred))
                test_mae = mae_test.append(
                    mean_absolute_error(y_test, y_pred_random_forest))
                print(y_test, y_pred_random_forest)
                accuracy_of_random_forest = metrics.accuracy_score(
                    y_test, y_pred_random_forest)
                print("accuracy:\t")

            RanFor()

            def svc():
                support_vc = SVC()
                support_vc.fit(x_train, y_train)
                y_pred_support_vc = support_vc.predict(x_test)
                y_train_pred = support_vc.predict(x_train)
                mae_train = []
                mae_test = []
                train_mae = mae_train.append(
                    mean_absolute_error(y_train, y_train_pred))
                test_mae = mae_test.append(
                    mean_absolute_error(y_test, y_pred_support_vc))
                print(y_test, y_pred_support_vc)
                accuracy_of_svc = metrics.accuracy_score(
                    y_test, y_pred_support_vc)
                print("accuracy:\t")
            svc()
        model_for_categorical_data()
    else:
        diff = {}
       
        def model_for_numerical_data():

            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.svm import SVR
            from sklearn import metrics
            from sklearn.metrics import mean_absolute_error
            import matplotlib.pyplot as plt
            # def standardization_and_normalization(df):

            def Knr():
                knr = KNeighborsRegressor(n_neighbors=3)
                knr.fit(x_train, y_train)
                y_pred_knr = knr.predict(x_test)
                y_train_pred = knr.predict(x_train)
                print(x_test.shape, x_train.shape, y_test.shape,
                      y_train.shape, y_pred_knr.shape, y_train_pred.shape)
                mae_train = []
                mae_test = []
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_pred_knr)
                diff['Knr'] = test_mae-train_mae
                print(diff['Knr'])
                global graphA
                graphA = gf.graphToImg(
                    (y_pred_knr), y_test, 'blue', y_train, y_train_pred, 'pink', 'KNN','y_pred_knn', 'y_test')
                global accuracy
                accuracy = [diff['Knr']]
                print('accuracy is :{}', accuracy)

            Knr()

            def LinReg():
                linear = LinearRegression()
                linear.fit(x_train, y_train)
                y_pred_linear = linear.predict(x_test)
                y_train_pred = linear.predict(x_train)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_pred_linear)
                diff['LinReg'] = test_mae-train_mae
                print(diff['LinReg'])
                global graphB
                graphB = gf.graphToImg(
                    np.array(y_pred_linear), y_test, 'red', y_train, y_train_pred, 'green','Linear Regression','y_train','y_train_pred')
                accuracy.append(diff['Knr'])
                print('accuracy is :{}', accuracy)

                # plotting the graph
                # plt.scatter(np.array(y_pred_linear),y_test,c='blue')
                # plt.scatter(y_train,y_train_pred,c='pink')
                # plt.show()
            LinReg()

            def DeciTreeReg():
                dectree = DecisionTreeRegressor()
                dectree.fit(x_train, y_train)
                y_pred_dectree = dectree.predict(x_test)
                y_train_pred = dectree.predict(x_train)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_pred_dectree)
                diff['DeciTreeReg'] = test_mae-train_mae
                print(diff['DeciTreeReg'])
                global graphC
                graphC = gf.graphToImg(
                    np.array(y_pred_dectree), y_test, 'black', y_train, y_train_pred, 'yellow','DecisionTree','y_pred_dectree','y_test')
                accuracy.append(diff['Knr'])
                print('accuracy is :{}', accuracy)
                # plotting the graph
                # plt.scatter(np.array(y_pred_dectree),y_test,c='blue')
                # plt.scatter(y_train,y_train_pred,c='pink')
                # plt.show()

            DeciTreeReg()

            def RanForReg():
                random_forest = RandomForestRegressor()
                random_forest.fit(x_train, y_train)
                y_pred_random_forest = random_forest.predict(x_test)
                y_train_pred = random_forest.predict(x_train)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_pred_random_forest)
                diff['RanForReg'] = test_mae-train_mae
                print(diff['RanForReg'])

                global graphD
                graphD = gf.graphToImg(np.array(
                    y_pred_random_forest), y_test, 'violet', y_train, y_train_pred, 'orange','RandomForest','y_pred','y_test')
                accuracy.append(diff['Knr'])
                print('accuracy is :{}', accuracy)
                # plotting the graph
                # plt.scatter(np.array(y_pred_random_forest),y_test,c='blue')
                # plt.scatter(y_train,y_train_pred,c='pink')
                # plt.show()
            RanForReg()

            def SVReg():
                support_vr = SVR()
                support_vr.fit(x_train, y_train)
                y_pred_support_vr = support_vr.predict(x_test)
                y_train_pred = support_vr.predict(x_train)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_pred_support_vr)
                diff['SVReg'] = test_mae-train_mae
                print(diff['SVReg'])
                global graphE
                graphE = gf.graphToImg(
                    np.array(y_pred_support_vr), y_test, 'black', y_train, y_train_pred, 'orange','SupportVector','y_pred_sv','y_test')
                accuracy.append(diff['Knr'])
                print('accuracy is :{}', accuracy)
                # plotting the graph
                # plt.scatter(np.array(y_pred_support_vr),y_test,c='blue')
                # plt.scatter(y_train,y_train_pred,c='pink')
                # plt.show()

            SVReg()
            return df
        model_for_numerical_data()


def standardization(df):
    def min_max_scaling():
        df_min_max_scaled = df.copy()
        df_columns = df_min_max_scaled.columns.values.tolist()
        print(df_columns)
        for i in range(0, len(df_columns)):
            print(i)
            df_min_max_scaled[df_columns[i]] = (df_min_max_scaled[df_columns[i]] - df_min_max_scaled[df_columns[i]].min()) / (
                df_min_max_scaled[df_columns[i]].max() - df_min_max_scaled[df_columns[i]].min())
        return df_min_max_scaled
    df1 = min_max_scaling()
    return df1
# machine_learning_model(data,output_column)
