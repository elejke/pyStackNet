import os
import re
import subprocess
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class StackNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklean-like wrapper on the StackNet implemented by kaz-Anova:
        https://github.com/kaz-Anova/StackNet
    """
    def __init__(self, 
                 data_home,
                 stacknet_home,
                 n_folds=10,
                 n_jobs=10,
                 stack_data=False,
                 random_state=2017,
                 metric='logloss',
                 verbose=True,
                 params_file='params.txt',
                 train_file="train_stacknet.csv", 
                 test_file="test_stacknet.csv"):
        """
        Scikit-learn like wrapper on the StackNet implemented by kaz-Anova:
        https://github.com/kaz-Anova/StackNet
        """
        super().__init__()
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_folds = n_folds
        self.data_home = data_home            
        self.stacknet_home = stacknet_home
        self.stack_data = stack_data
        self.verbose = verbose
        self.train_file = train_file
        self.test_file = test_file
        self.params = params_file
        # Object dependent variables:
        self._train_file = os.path.join(self.data_home, self.train_file)
        self._test_file = os.path.join(self.data_home, self.test_file)
        self._output = None
        self._X = None
        self._y = None
        
    def fit(self, X, y):
        """
        Fitting the StackNet model:
        Arguments:
            
        Return:
        """
        self._X = X
        self._y = y
        return self
    
    def predict(self, X_test):
        """
        Predict method:
        """
        y_pred = np.argmax(self.predict_proba(X_test), axis=1)
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict and fit the model at the moment:
        FUCKING PROBLEMS WITH JAVA.
        """
        self._data_preproc(self._X, self._y, X_test)
        # Execute the StackNet.jar model and parse the results: 
        self._output = str(self._exec_stacknet())
        if self.verbose > 0:
            print(self._output)
        y_proba = pd.read_csv(os.path.join(self.stacknet_home, "stacknet_pred.csv"), header=None)
        
        return y_proba.values
    
    def get_scores(self):
        """
        Return scores from the folds:
        """
        self._scores = re.findall("[0-9].[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]+", self._output)
        self._scores = np.array(self._scores).astype(float)
        return self._scores
        
        
    def _data_preproc(self, X, y, X_test, y_test=None):
        """
        Preprocessing the data for StackNet. 
        Scaling add the y to X and store in DATA_HOME
        """
        
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
        y_test = np.array(y_test)    

        # y need to be a column:
        if y.shape == y.flatten().shape:
            y = y.reshape(-1, 1)

        # Scale the data
        stda = StandardScaler()
        stda.fit(np.vstack([X, X_test]))

        X_test = stda.transform(X_test)
        X = stda.transform(X)

        # Stack target to X (train)
        X = np.column_stack((y, X))

        # Stack id to X_test
        #X_test = np.column_stack((ids, X_test))

        # Export to txt files (, del.)
        np.savetxt(self._train_file, X, delimiter=",", fmt='%.5f')
        np.savetxt(self._test_file, X_test, delimiter=",", fmt='%.5f')
        
    def _exec_stacknet(self):
        """
        Executing the stacknet model and get the predictions:
        """
        # Saving current directory
        current_dir = os.getcwd()
        os.chdir(self.stacknet_home)
        # Commandline msg for StackNet.jar
        command = "java"
        command += " -Xmx3048m -jar " + os.path.join(self.stacknet_home, "StackNet.jar")
        command += " train task=classification"
        command += " train_file=" + self._train_file #train_stacknet.csv"
        command += " test_file=" + self._test_file #test_stacknet.csv"
        command += " model_file=" + os.path.join(self.stacknet_home, "stacknet.model") #stacknet.model"
        command += " params=" + self.params# "params.txt"
        command += " test_target=false"
        command += " pred_file=" + os.path.join(self.stacknet_home, "stacknet_pred.csv")
        command += " verbose=true"
        command += " Threads=" + str(self.n_jobs) #10"
        command += " stackdata=" + str(self.stack_data).lower() #false"
        command += " folds=" + str(self.n_folds) #10" 
        command += " seed=" + str(self.random_state) #2017"
        command += " metric=" + str(self.metric) #logloss"
        # Execute the command to train and predict_proba via StackNet.jar
        verbose_msg = subprocess.getoutput(command)#, shell=True)
        # Change working directory to the previous one
        os.chdir(current_dir)
        return verbose_msg