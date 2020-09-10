import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.2f}".format

import sys

sys.path.insert(1, '../fuzzylearn/')

from fuzzylearn import *
from fuzzylearn.fuzzifiers import LinearFuzzifier, CrispFuzzifier,ExponentialFuzzifier,QuantileLinearPiecewiseFuzzifier, QuantileConstantPiecewiseFuzzifier
from fuzzylearn.kernel import GaussianKernel, LinearKernel, HyperbolicKernel, PolynomialKernel, HomogeneousPolynomialKernel
from fuzzylearn import solve_optimization_gurobi

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, RobustScaler, PowerTransformer, Normalizer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score

import json
import logging

def addestra(model_class, X, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = None, scaling=StandardScaler(), dim_reduction=None):
        
    if logger is not None:
        logger.info('Inizio Addestramento')
        
    X_std = scaling.fit_transform(X)  if scaling is not None else X
    
    X_std = dim_reduction.fit_transform(X_std) if dim_reduction is not None else X_std
    
    clf = GridSearchCV(estimator=model_class(), param_grid=model_selection_grid, cv=num_fold_grid_search, n_jobs=-1)
    clf.fit(X_std,y)
    
    grid = clf.best_params_
    for i, j in grid.items():
        grid[i] = str(j)

    val = cross_val_score(clf, X_std, y, cv=num_fold_cross_val)
    
    if logger is not None:
        logger.info('%s',str(clf.best_params_))
        logger.info('Scaler: %s', str(scaling))
        logger.info('Scores: %s',str(val))
        logger.info("Fine addestramento")
        
    return val, grid
    
def esperimento(dataset,columns,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = None, scaling=StandardScaler(), dim_reduction=None,label=None):
    grid = {}
    if logger is not None:
        name = label if label is not None else str(columns)
        logger.info('ESPERIMENTO CON %s',name)
        
    dataset_values = dataset[columns].values
    result, grid = addestra(FuzzyInductor,dataset_values,y,model_selection_grid,num_fold_grid_search,num_fold_cross_val,dim_reduction=dim_reduction,scaling=scaling,logger = logger)
    
    if logger is not None:
        logger.info('Score: %s',str(result.mean()))
        logger.info("FINE ESPERIMENTO\n")
    
    return result.mean(), grid
    
def esperimento_registrato(dataset,columns,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = None, file_json = None, scaling=StandardScaler(), dim_reduction=None,label=None):
    grid = {}
    sd_index = label if label is not None else str(columns)
    if file_json is not None:
        with open(file_json, "r") as read_file:
            esperimenti = json.load(read_file)
    else:
        esperimenti = {}
    if sd_index not in esperimenti.keys():
        score, grid = esperimento(dataset,columns,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = logger, scaling=scaling, dim_reduction=dim_reduction,label=sd_index)
            
        esperimenti[sd_index] = grid
        esperimenti[sd_index]['score'] = score
        if file_json is not None:
        	with open(file_json, "w") as write_file:
            		json.dump(esperimenti, write_file)
    col = pd.DataFrame(index=esperimenti[sd_index].keys())
    col[sd_index] = esperimenti[sd_index].values()
    return col
    
def fuzzifier_table(dataset,fuzzifiers,columns,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = None, scaling=StandardScaler(), dim_reduction=None,label=None):
    sd_index = label if label is not None else str(columns)
    table = pd.DataFrame({sd_index:[None for f in fuzzifiers]},index=[fuzzifiers_to_string[f] for f in fuzzifiers])

    X = dataset[columns].values
    
    X_std = scaling.fit_transform(X)  if scaling is not None else X
    
    X_std = dim_reduction.fit_transform(X_std) if dim_reduction is not None else X_std
    
    clf = GridSearchCV(estimator=model_class(), param_grid=model_selection_grid, cv=num_fold_grid_search, n_jobs=-1)
    clf.fit(X_std,y)
    
    bs_grid = clf.best_params_
    
    for i,j in bs_grid.items():
        bs_grid[i] = [j]
    
    for fuzzifier in fuzzifiers:
        params_fuzzifier = bs_grid
        params_fuzzifier['fuzzifier'] = [fuzzifier]
        score, _ = esperimento(dataset,columns,model_class, y, params_fuzzifier, num_fold_grid_search, num_fold_cross_val, logger = logger, scaling=scaling, dim_reduction=dim_reduction,label=sd_index)
        table.loc[fuzzifiers_to_string[fuzzifier]][sd_index] = score
    return table
    
    
def incidenti_fuzzifier_table(dataset,subdatasets,fuzzifiers,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = None, scaling=StandardScaler(), dim_reduction=None, file_json=None,labels=None):
    table = pd.read_json(file_json) if file_json is not None else pd.DataFrame(index=[fuzzifiers_to_string[f] for f in fuzzifiers])
    if len(table.index) == 0:
        table = pd.DataFrame(index=[fuzzifiers_to_string[f] for f in fuzzifiers])
    df_cols = labels if labels is not None else subdatasets
    for col,sd in zip(df_cols,subdatasets):
        if str(col) not in table.columns:
            table[str(col)] = fuzzifier_table(dataset,fuzzifiers,sd,model_class, y, model_selection_grid, num_fold_grid_search, num_fold_cross_val, logger = logger, scaling = scaling,dim_reduction=dim_reduction,label=str(col))
            if file_json is not None:
                 with open(file_json, "w") as write_file:
                    json.dump(table.to_dict(), write_file)
    return table
    
fuzzifiers_class = [LinearFuzzifier,ExponentialFuzzifier, CrispFuzzifier, QuantileConstantPiecewiseFuzzifier, QuantileLinearPiecewiseFuzzifier]
fuzzifiers_to_string = {LinearFuzzifier: "LinearFuzzifier",
                        ExponentialFuzzifier:"ExponentialFuzzifier",
                        CrispFuzzifier: "CrispFuzzifier",
                        QuantileConstantPiecewiseFuzzifier: " QuantileConstantPiecewiseFuzzifier",
                        QuantileLinearPiecewiseFuzzifier: "QuantileLinearPiecewiseFuzzifier"}
                        
def classify(item, est_mu,classes):
    return sorted([(l, est_mu[l](item)) for l in classes], key=lambda i:i[1], reverse=True)[0][0]
    
def to_membership_values(labels, target):
    return [1 if l==target else 0 for l in labels]
    
def best_estimator_holdout(best_estimators,data_index,data_values, data_labels,classes,train_percentage,classifier, num_holdouts):
    n = len(data_values)
    
    performance_train = performance_test = 0.0
    
    trials = range(num_holdouts)
    
    for trial in trials:
    
        permuted_indices = np.random.permutation(n)
        train_indices = permuted_indices[:int(n*train_percentage)]
        test_indices = permuted_indices[int(n*train_percentage):]
    
        train_values = [data_values[i] for i in train_indices]
        test_values = [data_values[i] for i in test_indices]
        
        train_values_arr = np.asarray(train_values)
        
        mu = {} 
        mu_train = {}
        mu_test = {}
        
        result = []
        
        for target in classes:
            mu[target] = to_membership_values(data_labels, target)
            mu_train[target] = [mu[target][i] for i in train_indices]
            mu_test[target] = [mu[target][i] for i in test_indices]
        
        labels_train = [data_labels[i] for i in train_indices]
        labels_test = [data_labels[i] for i in test_indices]
        
        est_mu = {}
        
        targets = classes
        
        for target,be in zip(targets,best_estimators):
            be.fit(train_values_arr, mu_train[target], n_jobs=-1)
            est_mu[target] = be.estimated_membership_

            if len(est_mu.values()) < len(classes):
                continue # at least one class not inferred
        
        index_train = [data_index[i] for i in train_indices]
        index_test = [data_index[i] for i in test_indices]
        
        results_train = list(zip(index_train,
                            map(lambda item: classifier(item, est_mu,classes),
                            train_values), labels_train))
        performance_train += 1.0 * len([r for r in results_train if r[1] != r[2]]) / len(results_train)
    
        results_test = list(zip(index_test,
                           map(lambda item: classifier(item, est_mu,classes),
                           test_values), labels_test))
    
        performance_test += 1.0 * len([r for r in results_test if r[1] != r[2]]) / len(results_test)
        
    return (performance_train/num_holdouts, performance_test/num_holdouts)

