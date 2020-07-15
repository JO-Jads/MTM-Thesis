import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
hyperparameter_settings=pd.read_json(r'pipelines.json')#hyperparameter settings
fn_data_feats = 'data_feats_featurized.csv' #dataset features
data_set_pipelines=pd.read_csv('all_normalized_accuracy_with_pipelineID.csv')

algorithms=['extra_trees', 'k_nearest', 'xgradient_boosting', 'quadratic_discriminant_analysis', 
            'bernoulli', 'gradient_boosting', 'decision_tree', 'multinomial', 'lda', 'random_forest'] #these are the different algorithms analyzed 
algarithm_dictionary={'pre_processing':['polynomial_include_bias', 'polynomial_interaction_only',
       'polynomial_degree','pca_keep_variance', 'pca_whiten'],'extra_trees':['extra_trees_bootstrap',
       'extra_trees_min_samples_leaf', 'extra_trees_n_estimators',
       'extra_trees_max_features', 'extra_trees_min_weight_fraction_leaf',
       'extra_trees_criterion', 'extra_trees_min_samples_split',
       'extra_trees_max_depth'],'id': ['id'],'k_nearest': ['k_nearest_neighbors_p', 'k_nearest_neighbors_weights',
       'k_nearest_neighbors_n_neighbors'],'xgradient_boosting':['xgradient_boosting_reg_alpha', 'xgradient_boosting_colsample_bytree',
       'xgradient_boosting_colsample_bylevel','xgradient_boosting_scale_pos_weight','xgradient_boosting_learning_rate','xgradient_boosting_max_delta_step',
       'xgradient_boosting_base_score', 'xgradient_boosting_n_estimators',
       'xgradient_boosting_subsample', 'xgradient_boosting_reg_lambda',
       'xgradient_boosting_min_child_weight', 'xgradient_boosting_max_depth',
       'xgradient_boosting_gamma'],'libsvm_svc':['libsvm_svc_kernel', 'libsvm_svc_C',
       'libsvm_svc_max_iter', 'libsvm_svc_degree', 'libsvm_svc_coef0',
       'libsvm_svc_tol', 'libsvm_svc_shrinking', 'libsvm_svc_gamma'],'quadratic_discriminant_analysis':['qda_reg_param'],'bernoulli':['bernoulli_nb_alpha','bernoulli_nb_fit_prior'],'gradient_boosting':['gradient_boosting_loss',
       'gradient_boosting_max_leaf_nodes', 'gradient_boosting_learning_rate',
       'gradient_boosting_min_samples_leaf', 'gradient_boosting_n_estimators',
       'gradient_boosting_subsample',
       'gradient_boosting_min_weight_fraction_leaf',
       'gradient_boosting_max_features', 'gradient_boosting_min_samples_split',
       'gradient_boosting_max_depth'],'decision_tree':['decision_tree_splitter',
       'decision_tree_max_leaf_nodes', 'decision_tree_min_samples_leaf',
       'decision_tree_max_features', 'decision_tree_min_weight_fraction_leaf',
       'decision_tree_criterion', 'decision_tree_min_samples_split',
       'decision_tree_max_depth'], 'multinomial':['multinomial_nb_alpha',
       'multinomial_nb_fit_prior'], 'lda':['lda_shrinkage', 'lda_shrinkage_factor',
       'lda_n_components', 'lda_tol'],'random_forest':['random_forest_bootstrap', 'random_forest_max_leaf_nodes',
       'random_forest_min_samples_leaf', 'random_forest_n_estimators',
       'random_forest_max_features', 'random_forest_min_weight_fraction_leaf',
       'random_forest_criterion', 'random_forest_min_samples_split',
       'random_forest_max_depth']}  #this dictionary contains all of the different hp from the different algorithms. 


dataset_list_overall=[10, 1004, 1020, 1459, 1460, 1464, 1467, 1471, 1472, 1473, 1475, 1481, 1482, 1021, 1486, 1487, 1488, 1489, 1496, 1498,
                      1500, 1501, 1507, 1508, 1022, 151, 1510, 1512, 1513, 1516, 1517, 1518, 1519, 1520, 1527, 1025, 1528, 1529, 1530,1531,
                      1532, 1533, 1534, 1535, 1536, 1537, 1026, 1538, 1539, 1540, 1541, 1546, 1556, 1557, 1561, 1563, 1564, 1036, 
                      1565,1568,16, 1600, 164, 18, 180, 181, 183, 1038, 184, 187, 189, 197, 20, 209, 21, 22, 223, 225, 1040, 227, 23, 230, 
                      26, 275,276, 277, 278, 279,1467, 1471, 1472, 1473, 1475, 1481, 1482, 1021, 1486, 1487, 1488, 1489, 1496, 1498,1500, 
                      1501, 1507, 1508,1022, 151, 1510, 1512, 1513, 1516, 1517, 1518, 1519, 1520, 1527, 1025, 1528, 1529, 1530, 
                      1531]#trainings datasets


data_set_pipelines['Unnamed: 0'].tolist()#Transforming ID
data_set_pipelines.columns.tolist()[1:]
a_file = open("data.pkl", "rb")#random selected pipelines file
random_pipelines_Dic = pickle.load(a_file)
id_dataset=list(random_pipelines_Dic.keys()) #dataset ids
min_max_scaler = preprocessing.MinMaxScaler()
temp_list=dataset_list_overall + id_dataset
for element in temp_list:#reshaping into scores between 0 and 1
    data_set_pipelines[[str(element)]] = min_max_scaler.fit_transform(data_set_pipelines[[str(element)]])

def creat_train_test_split(dataset_input):
    """
    This function returns the train/test from the test_dataset_input
    """
    df = data_set_pipelines
    pipeline_ids = df['Unnamed: 0'].tolist()
    dataset_ids = df.columns.tolist()[1:]
    dataset_ids = [int(dataset_ids[i]) for i in range(len(dataset_ids))]
    Y = df.values[:,1:].astype(np.float64)
    ids_train = dataset_list_overall #train dataset
    ids_test = list([dataset_input]) # test dataset
    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]
    Ytrain = Y[:, ix_train]
    Ytest = Y[:, ix_test]
    df = pd.read_csv(fn_data_feats)
    dataset_ids = df[df.columns[0]].tolist()
    ix_train = [dataset_ids.index(i) for i in ids_train]
    ix_test = [dataset_ids.index(i) for i in ids_test]
    Ftrain = df.values[ix_train, 1:]
    Ftest = df.values[ix_test, 1:]

    return Ytrain, Ytest, Ftrain, Ftest



def random_search(random_pipelines,bo_n_iters, ytest, speed=1, do_print=False):
    """
    Speed stands for how many random queries are performed per iteration. The bo_n_iters stands for the number of
    iterations and random_pipelines are the selected pipelines provided by the mtm model.
    """
    count=0
    ix_evaled = []
    ix_candidates = list(random_pipelines.index)
    ybest_list = []
    ybest = np.nan
    for l in range(bo_n_iters):
        for ll in range(speed):
            ix = ix_candidates[np.random.permutation(len(ix_candidates))[0]]
            if np.isnan(ybest):
                ybest = ytest[ix]
            else:
                if ytest[ix] > ybest:
                    ybest = ytest[ix]
            ix_evaled.append(ix)
            ix_candidates.remove(ix)
        ybest_list.append(ybest)
    return np.asarray(ybest_list)


def per_algorithm():
    '''
    This function is an algorithm-based method to obtain the random search results. 
    '''
    over_all_result_dictionary={}
    round_count=1
    for element in list(random_pipelines_Dic.keys()):
        Ytrain, Ytest, Ftrain, Ftest = creat_train_test_split(element)
        temp_dic={}
        for algorithm in algorithms:
            result_dictionary=[]
            for keys in random_pipelines_Dic[element].keys():
                random_pipelines=random_pipelines_Dic[element][keys][algorithm]
                regrets_random1x = np.zeros((int(keys)+1, Ytest.shape[1]))
                count=0
                for d in np.arange(Ytest.shape[1]):
                    count+=1
                    ybest = np.nanmax(Ytest[:,d])
                    result_dictionary.append(round(list(sorted(random_search(random_pipelines,int(keys)+1,Ytest[:,d], speed=1)))[-1],3))
                    regrets_random1x[:,d] = ybest - random_search(random_pipelines,int(keys)+1,Ytest[:,d], speed=1)
                round_count+=1
            temp_dic[algorithm]=result_dictionary
        over_all_result_dictionary[element]=temp_dic
            
    return over_all_result_dictionary

result=per_algorithm()

def transform_to_algorithm_average(dic):
    '''
    This function takes the dictionary with results per datasets and
    returns an algorithm-based dataset with an average score.
    '''
    dic_overall={}
    for algorithm in algorithms:
        score_list=[]
        for element in list(range(10)):
            empty_list=[]
            for keys in result.keys():
                empty_list.append(result[keys][algorithm][element])
            average=sum(empty_list)/len(empty_list)
            score_list.append(average)
        dic_overall[algorithm]=score_list
    return dic_overall

dic_overall=transform_to_algorithm_average(result)

a_file = open("random_search_results.pkl", "wb")
pickle.dump(dic_overall, a_file)
a_file.close()
