#Library in Model Selection class
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,recall_score,precision_score,f1_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC, NuSVC
from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
from xgboost              import XGBClassifier
from catboost             import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import RepeatedStratifiedKFold,RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNetCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error, max_error, mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import SGDRegressor, HuberRegressor,TheilSenRegressor, RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from imblearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from .Statistical_test import statical_test


class model_selection:
    """
    #Chức năng: chạy sơ bộ qua các model để chọn model phù hợp tối ưu.

    #Đầu vào: Data_train và Data_test hoàn chỉnh.

    #Đầu ra: kết quả sơ bộ đánh giá nội của từng model
    """
    def __init__(self, X_train, y_train,task = "C", score = "f1"):
        self.task = task
        self.scoring = score
        self.X_train = X_train
        self.y_train = y_train
        self.results = list()
        self.names = list()
        self.case = len(np.unique(self.y_train))

     # 1. Choose Regression or Classification
     #Case = 2: Classification
     # Case > 2: Regression
    def case_model(self):
        if self.task == "C":
            self.models,  self.names = self.Model_Classification()
            # define evaluation procedure
            self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

        elif self.task =="R":
            self.models,  self.names = self.Model_Regression()
            # define evaluation procedure
            self.cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)


    # 2. Model Regression
    def Model_Regression(self):
        models, names = list(), list()

        #1. LR
        model = LinearRegression()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('LR')

        #2. Ridge
        model = Ridge(alpha = 1, random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Ridge')

        #3. ElasticNet
        model = ElasticNetCV(cv = 5, random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('ELN')

        #4. HuberRegressor
        model = HuberRegressor()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Huber')

        #5. pcr
        models.append(make_pipeline(StandardScaler(), PCA(n_components=40), LinearRegression()))
        names.append('PCR')

        #6. PLS
        model = PLSRegression(n_components=40)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('PLS')

        #7. GPR
        kernel = DotProduct() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=kernel,random_state=42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('GPR')


        #8 KNN
        model = KNeighborsRegressor()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('KNN')

        #9 svm
        model = SVR(kernel='rbf', gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('SVM')

        #13 DT
        #model = DecisionTreeRegressor(random_state=42)
        #steps = [('m', model)]
        #models.append(Pipeline(steps=steps))
        #names.append('DTree')

        # 10 RF
        model = RandomForestRegressor(random_state=42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('RF')

        # 11 ExT
        model = ExtraTreesRegressor(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('ExT')

        #16 ADA
        #model = AdaBoostRegressor(random_state = 42)
        #steps = [('m', model)]
        #models.append(Pipeline(steps=steps))
        #names.append('ADA')

        #12 Grad
        model = GradientBoostingRegressor(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Grad')

        #13 XGB
        model = XGBRegressor(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('XGB')

        #14 Cat
        model = CatBoostRegressor(verbose = 0, random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Catbst')

        #15 Cat
        model = HistGradientBoostingRegressor(random_state = 42, verbose = 0)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Hist')

        #16 MLP
        model = MLPRegressor(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('MLP')
        return models, names

    # 3. Model Classification
    def Model_Classification(self):
        models, names = list(), list()

        #0. Dummy
        #model = DummyClassifier(strategy='stratified', random_state =42)
        #steps = [('m', model)]
        #models.append(Pipeline(steps=steps))
        #names.append('Baseline')

        #1. Logistics
        model = LogisticRegression(penalty = 'l2', max_iter = 10000)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Logic')

        #2 KNN
        model = KNeighborsClassifier(n_neighbors = 10)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('KNN')

        #3 svm
        model = SVC(probability = True, max_iter = 1000)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('SVM')

        #4 GaussianNB
        model = GaussianNB()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('GaussianNB')

        #5 BernoulliNB
        model = BernoulliNB()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('BernoulliNB')

        #6. LDA
        model = LinearDiscriminantAnalysis()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('LDA')

        #7. QDA
        model = QuadraticDiscriminantAnalysis()
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('QDA')

        #8 DT
        model = DecisionTreeClassifier(ccp_alpha=.02)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('DTree')

        #9 RF
        model = RandomForestClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('RF')

        #10 ExT
        model = ExtraTreesClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('ExT')

        #11 ADA
        model = AdaBoostClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('ADA')

        #12 Grad
        model = GradientBoostingClassifier(random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Grad')

        #13 XGB
        model = XGBClassifier(random_state = 42, verbosity=0, use_label_encoder=False, eval_metrics ='logloss')
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('XGB')

        #14 Cat
        model = CatBoostClassifier(verbose = 0, random_state = 42)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('Catbst')

        #15 MLP
        model = MLPClassifier(alpha = 0.01, max_iter = 10000, validation_fraction = 0.1, random_state = 42, hidden_layer_sizes = 150)
        steps = [('m', model)]
        models.append(Pipeline(steps=steps))
        names.append('MLP')
        return models, names

    def evaluate_model(self, X, y, model):
      #Kết quả đánh giá model
        scores = cross_val_score(model, X, y, scoring=self.scoring, cv=self.cv, n_jobs=-1)
        return scores

    def compare(self):
      # Đưa tất cả các kết quả đánh giá vào list_results
        for i in range(len(self.models)):
            scores = self.evaluate_model(self.X_train, self.y_train, self.models[i])
            self.results.append(scores)
            print('>%s %.3f (%.3f)' % (self.names[i], np.mean(scores), np.std(scores)))
        self.model_compare = statical_test(self.results, self.names,X_train = self.X_train, y_train = self.y_train)
        self.model_compare.visualize()

    def visualize(self):
      #Vẽ biểu đồ boxplot thể hiện hiệu quả của từng model, từ đó chọn model phù hợp để tối ưu.
        mean = list()
        for i in range (len(self.results)):
            x = self.results[i].mean().round(3)
            mean.append(x)
        data = np.array(mean)
        ser = pd.Series(data, index =self.names)


        dict_columns = {'Mean':mean,'Method':self.names,}
        df = pd.DataFrame(dict_columns)


        sns.set_style("whitegrid")
        plt.figure(figsize=(50,20))
        box_plot = sns.boxplot(data=self.results,showmeans=True ,meanprops={"marker":"d",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"})
        box_plot.axes.set_title("Model Compare", fontsize=16)
        box_plot.set_xlabel("Method", fontsize=14)
        box_plot.set_ylabel("Results", fontsize=14)
        vertical_offset = df["Mean"].median()*0.01

        for xtick in box_plot.get_xticks():
            box_plot.text(xtick,ser[xtick]+ vertical_offset,ser[xtick],
            horizontalalignment='center',size='x-large',color='w',weight='semibold')

        box_plot.get_xticks(range(len(self.results)))
        box_plot.set_xticklabels(self.names, rotation='horizontal')