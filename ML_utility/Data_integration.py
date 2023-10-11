import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

class Data_Itegration():
    """
    Create Data Frame from csv file, find missing value (NaN), choose a threshold to make target transformation (Classification)
    remove handcrafted value (Regression), split Data to Train and Test and show the chart

    Input:
    ------
    file csv, activity threshold

    Returns:
    --------
    Raw data_train and data_test, bar chart showing imbalanced data.

    """
    def __init__(self,col_drop, threshold, y_name, path):
        self.col_drop = col_drop
        self.thresh = threshold
        self.Y_name = y_name
        self.path = path
        # while True:
        #   #Nhập đường dẫn file data
        #     self.path = data_path
        #     checkPath = os.path.isfile(self.path)
        #     if checkPath == True:
        #         break
        self.data = None
        #Đọc dữ liệu
        self.data_csv()
        #Chia data
        self.Data_split()

        #self.save_csv()

    # 1. Read Data
    def data_csv(self):
        self.data = pd.read_csv(self.path)
        display(self.data.head(5))
        self.data = self.data.drop([self.col_drop], axis = 1)
        display(self.data.head(2))

    # 2. Check nan value - Mark Nan value to np.nan
    #Dùng lệnh hàm for để check dữ liệu bị mất
    def Check_NaN(self, data):
        index = []
        for key, value in enumerate(data):
            if type(value) == float or type(value) == int:
                continue
            else:
                index.append(key)
        data[index] = np.nan

    # 3. Target transformation - Classification
    #Chuyễn dữ liệu của cột Y  về phân loại nhị phân (nhãn 0, 1) dựa vào mức hoạt tính phù hợp
    #Đầu vào: Data, mức hoạt tính.
    def target_bin(self, thresh):
        while True:
            try:
                input_target_style = 2
                break
            except:
                print("Error value")
        if input_target_style == 1:
            self.thresh = thresh
            t1 = self.data[self.Y_name] < self.thresh
            self.data.loc[t1, self.Y_name] = 1
            t2 = self.data[self.Y_name] >= self.thresh
            self.data.loc[t2, self.Y_name] = 0
            self.data[self.Y_name] = self.data[self.Y_name].astype('int64')
        else:
            self.thresh = thresh
            t1 = self.data[self.Y_name] < self.thresh
            self.data.loc[t1, self.Y_name] = 0
            t2 = self.data[self.Y_name] >= self.thresh
            self.data.loc[t2, self.Y_name] = 1
            self.data[self.Y_name] = self.data[self.Y_name].astype('int64')


    # 4. Remove handcrafted data - Regression - active <==> pChEMBL = 0;  non-active <==> pChEMBL =10
    def remove_handcrafted_data(self):
        idx = self.data.loc[(self.data[self.Y_name] ==0) | (self.data[self.Y_name] ==10)].index
        self.data.drop(idx, axis =0, inplace = True)

    # 5. Split data
    #Chia tập dữ liệu  thành tập train và test.
    def Data_split(self):

        self.data.apply(self.Check_NaN)
        col=np.array(self.data[self.Y_name], self.data[self.Y_name].dtype)
        self.data[self.Y_name]=col
        # while True:
        #     self.RoC = "Y"
        #     if self.RoC == 'Y' or self.RoC == 'N':
        #         break
        # if self.RoC.title() == "Y":
        #     while True:
        #         try:
        #             self.thresh = 
        #             break
        #         except:
        #             print("Error value!")
        self.target_bin(thresh = self.thresh)
        y = self.data[self.Y_name]
        self.stratify = y



        X = self.data.drop([self.Y_name], axis =1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42, stratify = self.stratify)


        #index:
        self.idx = X.T.index

        #Train:
        self.df_X_train = pd.DataFrame(X_train, columns = self.idx)
        self.df_y_train = pd.DataFrame(y_train, columns = [self.Y_name])
        self.Data_train = pd.concat([self.df_y_train, self.df_X_train], axis = 1)


        #test
        self.df_X_test = pd.DataFrame(X_test, columns = self.idx)
        self.df_y_test = pd.DataFrame(y_test, columns = [self.Y_name])
        self.Data_test = pd.concat([self.df_y_test, self.df_X_test], axis = 1)

        print("Data train:", self.Data_train.shape)
        print("Data test:", self.Data_test.shape)
        print(75*"*")
        display(self.Data_train.head(5))
        self.Visualize_target()



    def Visualize_target(self):
        sns.set('notebook')
        plt.figure(figsize = (16,5))
        plt.subplot(1,2,1)
        plt.title(f'Training data', weight = 'semibold', fontsize = 16)
        plt.hist(self.Data_train.iloc[:,0])
        plt.xlabel(f'Imbalanced ratio: {(round((self.Data_train.iloc[:,0].values == 1).sum() / (self.Data_train.iloc[:,0].values == 0).sum(),3))}')
        plt.subplot(1,2,2)
        plt.title(f'External validation data', weight = 'semibold', fontsize = 16)
        plt.hist(self.Data_test.iloc[:,0])
        plt.xlabel(f'Imbalanced ratio: {(round((self.Data_test.iloc[:,0].values == 1).sum() / (self.Data_test.iloc[:,0].values == 0).sum(),3))}')
        #plt.savefig("distribution.png", dpi = 600)
        plt.show()

        
#Library in Data preprocessing class
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
#from ML_utility.Data_integration import Data_Itergration

class Data_preprocess(Data_Itegration):
    """
    Preprocess data include:
     - Clean duplicated data (columns and rows)
     - Find percentage of missing value, choose suitable method to make imputation
     - Check variance threshold
     - Convert nomial to integer.

    Input:
    -----
    Raw Data_train and Data_test, variance threshold, imputation method.

    Returns:
    --------
    Data_train and Data_test

    """
    def __init__(self, Data_train, Data_test,y_name):
        self.data_train= Data_train.copy()
        self.data_test= Data_test.copy()
        #Chỗ này nên cho nhập để đỡ sửa code
        self.Y_name = y_name

    # 1. Remove duplicate data - rows and columns
    def Duplicate_data(self):
        # 1.1. Duplicate rows: xóa hàng bị trùng

        # train
        dup_rows = self.data_train.duplicated()
        print(f"Total duplicated rows-train: {(dup_rows == True).sum()}")
        print("Data train before drop duplicates:", self.data_train.shape[0])
        self.data_train.drop_duplicates(inplace = True)
        self.data_train.reset_index(drop = True, inplace = True)
        print("Data train after drop duplicates:", self.data_train.shape[0])
        print(75*"*")

        # test
        dup_rows = self.data_test.duplicated()
        print(f"Total duplicated rows-test: {(dup_rows == True).sum()}")
        print("Data test before drop duplicates:", self.data_test.shape[0])
        self.data_test.drop_duplicates(inplace = True)
        self.data_test.reset_index(drop = True, inplace = True)
        print("Data test after drop duplicates:", self.data_test.shape[0])
        print(75*"*")


        # 1.2. Duplicate columns: xóa cột bị trùng

        #Chuyển vị ma trận (Transpose)
        dup = self.data_train.T[self.data_train.T.duplicated()]
        idx = dup.index
        print(idx)


        print(f"Total similar columns: {dup.shape[0]}")

        #train
        print("Data train before drop duplicates:", self.data_train.shape)
        self.data_train.drop(idx, axis = 1, inplace = True)
        print("Data after drop duplicates:", self.data_train.shape)
        print(75*"*")

        #test
        print("Data test before drop duplicates:", self.data_test.shape)
        self.data_test.drop(idx, axis = 1, inplace = True)
        print("Data test after drop duplicates:", self.data_test.shape)
        print(75*"*")

    # 2. Check Variance Threshold:
    #Tìm ngưỡng phương sai phù hợp
    def Variance_Threshold(self):
        y = self.data_train[self.Y_name]
        X = self.data_train.drop([self.Y_name], axis =1)
        print(X.shape, y.shape)
        while True:
            try:
               # Define thresholds to check
                thresholds = np.arange(0.0, 1, 0.05)
                # Apply transform with each threshold
                results = list()
                for t in thresholds:
                # define the transform
                    transform = VarianceThreshold(threshold=t)
                # transform the input data
                    X_sel = transform.fit_transform(X)
                # determine the number of input features
                    n_features = X_sel.shape[1]
                    print('>Threshold=%.2f, Features=%d' % (t, n_features))
                # store the result
                    results.append(n_features)
                break
            except:
                break
        # plot the threshold vs the number of selected features
        plt.figure(figsize=(20,8))
        plt.title("Variance Threshold Analysis", fontsize = 24)
        plt.xlabel("Threshold", fontsize = 16)
        plt.ylabel("Number of Features", fontsize = 16)
        plt.plot(thresholds[:len(results)], results)
        plt.show()

    # 3. Remove variance Threshold:
    #Xóa dữ liệu có phương sai không phù hợp với ngưỡng đã chọn
    def Low_variance_cleaning(self):
        while True:
            try:
                thresh=0.05
                break
            except:
                print("Error threshold!")
        selector = VarianceThreshold(thresh)
        selector.fit(self.data_train)
        features = selector.get_support(indices = False)
        features[0]=True
        self.data_train = self.data_train.loc[:, features]
        lst = self.data_train.columns
        self.data_test = self.data_test.loc[:, features]
        print(75*"*")
###########################################################################################
# MISSING VALUES
    # 4. Find Missing Percentage:
    #Tìm hiểu tỉ lệ dữ liệu bị mất
    def find_missing_percent(self, data):
        """
        Returns dataframe containing the total missing values and percentage of total
        missing values of a column.
        """
        miss_df = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
        for col in data.columns:
            sum_miss_val = data[col].isnull().sum()
            percent_miss_val = round((sum_miss_val/data.shape[0])*100,2)
            missinginfo = pd.DataFrame({"ColumnName" : [col], "TotalMissingVals" : [sum_miss_val], "PercentMissing" : [percent_miss_val]})
            miss_df = pd.concat([miss_df,missinginfo])
            #miss_df = miss_df.append(missinginfo, ignore_index = True)

        miss_df = miss_df[miss_df["PercentMissing"] > 0.0]
        miss_df = miss_df.reset_index(drop = True)
        return miss_df

    # 5. Handle missing values
    #Xử lý dữ liệu missing bằng các phương pháp Imputation
    def Missing_value_cleaning(self):
        miss_df = self.find_missing_percent(self.data_train)
        print(miss_df)
        # Remove columns with high missing percentage
        while True:
            try:
                miss_thresh=0
                break
            except:
                print("Error threshold!")

        drop_cols = miss_df[miss_df['PercentMissing'] > miss_thresh].ColumnName.tolist()
        print("Drop_cols",  drop_cols)



        self.data_train.drop(drop_cols, axis =1, inplace = True)
        self.data_test.drop(drop_cols, axis =1, inplace = True)



        print("Total missing value-train", self.data_train.isnull().sum().sum())
        print("Total missing value-test", self.data_test.isnull().sum().sum())

        null_data_train = self.data_train[self.data_train.isnull().any(axis=1)]
        display(null_data_train)
        print("Total row-train with missing value", null_data_train.shape[0])

        null_data_test = self.data_test[self.data_test.isnull().any(axis=1)]
        display(null_data_test)
        print("Total row-test with missing value", null_data_test.shape[0])

        if null_data_train.shape[0] == 0:
            self.data_test = self.data_test.dropna(inplace = False)
            self.data_test.reset_index(drop = True, inplace = True)
        else:
            S = (input("Do you want to remove rows have missing values:").title())
            if S == 'Y':
                Data_train = self.data_train.dropna(inplace = False)
                Data_train.reset_index(drop = True, inplace = True)

                Data_test = self.data_test.dropna(inplace = False)
                Data_test.reset_index(drop = True, inplace = True)
            else:
                while True:
                    try:
                        M = int(input("Please select type of Imputation:\n\t1:Mean\n\t2:Median\n\t3:Mode\n\t4:KNNImputer\n\t5:MICE\n"))
                        break
                    except:
                        print("Error value!")
                if M == 1:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mean')
                elif M == 2:
                    imp=SimpleImputer(missing_vlues=np.NaN, strategy='median')
                elif M == 3:
                    imp=SimpleImputer(missing_values=np.NaN, strategy='mode')
                elif M == 4:
                    while True:
                        try:
                            n_neighbors = int(input("Please input n_neighbors:"))
                            break
                        except:
                            print("Error value!")
                    imp=KNNImputer(n_neighbors=n_neighbors)
                else:
                    A = int(input("Please select Algorithm of Imputation:\n\t1:BayesianRidge\n\t2:RandomForest\n"))
                    if A == 1:
                        estimator = BayesianRidge()
                    else:
                        estimator = GradientBoostingRegressor(random_state = 42)
                    imp= IterativeImputer(random_state=42, missing_values=np.NaN, estimator= estimator)

                imp.fit(self.data_train)



                # train
                Data_train=pd.DataFrame(imp.transform(self.data_train))
                Data_train.columns=self.data_train.columns
                Data_train.index=self.data_train.index
                display(Data_train.shape)
                # test
                Data_test=pd.DataFrame(imp.transform(self.data_test))
                Data_test.columns=self.data_test.columns
                Data_test.index=self.data_test.index
                display(Data_test.shape)
                print(75*"*")
            self.data_train = Data_train
            self.data_test = Data_test


    # 6. Convert low unique columns to inte
    #Chuyển dữ liệu về dạng số nguyên
    def Nomial(self):
        data = self.data_train.loc[:, (self.data_train.nunique() <10).values & (self.data_train.max() <10).values]  #feature with unique < 10 and max value <10 will set to be int64
        col = data.columns #select columns with int64
        #set all  col_olumn to int64
        print("DINH TINH:", col)

        self.data_train[col]=self.data_train[col].astype('int64')
        self.data_test[col]=self.data_test[col].astype('int64')
        display(self.data_train.head(5))



    # 7. Activate class
    def fit(self):
        self.Duplicate_data()
        self.Missing_value_cleaning() # missing values must be handled before feature selection
        self.Variance_Threshold()
        self.Low_variance_cleaning()
        features = list(self.data_train.columns)

        #print(features)
        self.Nomial()
        #self.save_csv()