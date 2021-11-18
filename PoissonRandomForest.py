class PoissonDecisionTree():

    def __init__(self):
        self

    from numpy.core.records import array

    def PoissonRandomForestRegressor (self, X_train:array, Y_train:array, X_test:array, params:dict, n: int)->list:
        """
        The function implements the Random Forest Regressor on a Poisson Bootstrap sample.
        """

        #Import libraries
        import itertools as it
        import pandas as pd
        import numpy as np
        from sklearn.tree import DecisionTreeRegressor

        # Function implementation of Poisson bootstrap sampling.
        def poiss(df):
            """
            Function implementation of Poisson bootstrap sampling.
            """
            poisson = np.random.poisson(size = len(df))
            new_df = []
            for ind, cnt in enumerate(poisson):
                if cnt != 0:
                    new_df += it.repeat(ind, cnt)
                else:
                    continue
            df_poiss = df.loc[new_df,:]
            return df_poiss

        # Connecting arrays to a DataFrame
        X, Y = pd.DataFrame(X_train), pd.DataFrame(Y_train)
        new_df = pd.concat([X,Y], axis=1)

        # The cycle of learning the DecisionTreeRegressor model and predicting the target variable of the test sample.
        best_models = []
        for _ in range(1, n+1):
            
            new_df_train = poiss(new_df)
            X, Y = new_df_train.iloc[:,:-1], new_df_train.iloc[:,-1]
            X, Y = X.to_numpy(), Y.to_numpy()
            clf = DecisionTreeRegressor(**params)
            clf.fit(X,Y)
            result = clf.predict(X_test)
            best_models.append(result)
            
        # Let's sum up all the elements of the list by index.
        results = np.asarray(best_models).sum(axis=0)

        # We determine the average value for each index of the array.
        fin_res = [j/n for j in results]
        return fin_res

    def PoissonRandomForestClassifier (self, X_train: array, Y_train: array, X_test: array, params: dict, n: int)->list:
        """
        The function implements the Random Forest Classifire on a Poisson Bootstrap sample.
        """
        
        #Import libraries
        import itertools as it
        import pandas as pd
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier
        
        # Function implementation of Poisson bootstrap sampling.
        def poiss(df):
            """
            Function implementation of Poisson bootstrap sampling.
            """
            poisson = np.random.poisson(size = len(df))
            new_df = []
            for ind, cnt in enumerate(poisson):
                if cnt != 0:
                    new_df += it.repeat(ind, cnt)
                else:
                    continue
            df_poiss = df.loc[new_df,:]
            return df_poiss

        # Connecting arrays to a DataFrame
        X, Y = pd.DataFrame(X_train), pd.DataFrame(Y_train)
        new_df = pd.concat([X,Y], axis=1)

        # The cycle of learning the DecisionTreeClassifire model and predicting the target variable of the test sample.
        best_models = []
        for _ in range(1, n+1):

            new_df_train = poiss(new_df)
            X, Y = new_df_train.iloc[:,:-1], new_df_train.iloc[:,-1]
            X, Y = X.to_numpy(), Y.to_numpy()
            clf = DecisionTreeClassifier(**params)
            clf.fit(X,Y)
            result = clf.predict(X_test)
            best_models.append(result)

        # Creating a dataframe from the predicted values.
        ddff = pd.DataFrame(best_models)

        # We determine the most frequent element in each feature.
        fin_res = [ddff[j].mode()[0] for j in ddff]
        return fin_res
