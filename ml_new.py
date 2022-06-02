import pandas as pd
import numpy as np
import logging as log
# data=pd.read_csv('customers.csv')
# print(data.columns)
output_column = ''
remove_columns = []
# output_column=input("enter output column name")
# for i in range(0,len(data.columns.values.tolist())):
# 	if output_column==data.columns.values.tolist()[i]:
# 		print(output_column)#1
# column data type check and convert it into numerical form if not
# check row size and apply restriction


def data_conversion_and_data_transformation(data):
    # df=pd.read_csv(data)#reading the
    # df
    global df
    df = pd.DataFrame(data)
    # print(df)
    for i in range(0, df.shape[1]):
        if df[df.columns[i]].dtypes == 'O':
            if (len(df[df.columns[i]].unique().tolist())) <= 5:
                k = 0
                t1 = df[df.columns[i]].unique().tolist()
                l = [
                    k+j for j in range(0, len(df[data.columns[i]].unique().tolist()))]
                # print(t1,l)
                df[df.columns[i]].replace(t1, l, inplace=True)

    def correlation(df, output_column):
        log.info('df is here :{}',df)
        corre = df.corr()[output_column]
        # print(corre)
        corr_column = corre.index.values
        # print(corr_column)
        corre = corre.tolist()
        # print(len(corre))
        dropped_column = []
        # for i in range(0, len(corre)):
        #     if ((corre[i] < 0.1) | (corre[i] == 1)) | (corre[i] > -0.1):
        #     # if (corre[i] < 0.2):
        #         dropped_column.append(corr_column[i])
            # print(dropped_column)
        # print(df)
        df=df.drop(axis=1, columns=dropped_column)
        return df
        # print(dropped_column)
    df=correlation(df, output_column)
    # print(df1.columns.values)
    # make custom button for user to take some info to drop some columns like IDs,roll_no,index,or something that only
    # shows serial no.
    # if df1.empty()==False :
    df.drop(columns=remove_columns)
    # print(df)
    return df
# data=data_conversion_and_data_transformation(data)


def null_value_count(df):
    # Data retrieval and conversion into dataframe
    # data conversion not done
    null_count = 0
    rows_to_be_dropped = []
    columns_to_be_dropped = []
    temp = 0
    for i in range(0, df.shape[0]):
        for j in range(0, df.shape[1]):
            if df.isnull().values[i][j] == True:
                null_count = null_count+1
                temp = temp+1
        if temp >= (0.7*df.shape[0]):  # criteria for dropping rows
            rows_to_be_dropped.append(i)
    for i in range(0, df.shape[1]):
        for j in range(0, df.shape[0]):
            if np.isnan(df.loc[j][df.columns[i]]) == True:
                temp = temp+1
        if temp >= (0.4*df.shape[0]):  # criteria for dropping columns
            columns_to_be_dropped.append(df.columns[i])
    # print(null_count)

    def null_column_dropping_and_dataset_category_and_null_value_filling(df, rows_to_be_dropped, columns_to_be_dropped):
        df.dropna(axis=0, subset=[rows_to_be_dropped], inplace=True)
        df.dropna(axis=1, subset=[columns_to_be_dropped], inplace=True)
    # here we have to check whether the columns values or categorical or not

        def null_value_filling(df):
            def outlier_detection_and_null_value_filling(df):
                outlier = []
                for i in range(0, df.shape[1]):
                    sort_data = np.sort(df[df.columns[i]])
                    Q1 = np.percentile(df, 25, interpolation='midpoint')
                    Q3 = np.percentile(df, 75, interpolation='midpoint')
                    IQR = Q3-Q1
                    low_lim = Q1 - 1.5 * IQR
                    up_lim = Q3 + 1.5 * IQR
                    column_with_null_and_outlier = []
                    for y in df[df.columns[i]].values:
                        if (y > up_lim) or (y < low_lim):
                            outlier.append(y)
                            column_with_null_and_outlier.append(df.columns[i])
                    if len(outlier) != 0:
                        for i in range(0, df.shape[0]):
                            for j in range(0, df.shape[1]):
                                if df.isnull().values[i][j]:
                                    # print(data.iloc[i,j])
                                    column_with_null_and_outlier.append(
                                        df.columns[j])
                        for k in range(0, len(column_with_null_and_outlier)):
                            df[column_with_null_and_outlier[k]] = df[column_with_null_and_outlier[k]].fillna(
                                df[column_with_null_and_outlier[k]].median())
            outlier_detection_and_null_value_filling(df)

            def categorical_or_numerical(df):
                column_with_null = []
                if df[output_column].unique().shape[0] > 5:
                    for i in range(0, df.shape[0]):
                        for j in range(0, df.shape[1]):
                            if df.isnull().values[i][j]:
                                # print(data.iloc[i,j])
                                column_with_null.append(df.columns[j])
                    for k in range(0, len(column_with_null)):
                        df[column_with_null[k]] = df[column_with_null[k]].fillna(
                            df[column_with_null[k]].mean())
                else:
                    for i in range(0, df.shape[0]):
                        for j in range(0, df.shape[1]):
                            if df.isnull().values[i][j]:
                                # pr(data.iloc[i,j])
                                column_with_null.append(df.columns[j])
                    for k in range(0, len(column_with_null)):
                        df[column_with_null[k]] = df[column_with_null[k]].fillna(
                            df[column_with_null[k]].mode())
            categorical_or_numerical(df)
        null_value_filling(df)
    null_column_dropping_and_dataset_category_and_null_value_filling(
        df, rows_to_be_dropped, columns_to_be_dropped)
    return df
# null_value_count(data)
