import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import random
import pickle
import json
import matplotlib.pyplot as plt
from ipywidgets import IntProgress
from IPython.display import display
from operator import itemgetter
from sklearn import preprocessing
random.seed(28)


data_set_pipelines=pd.read_csv('all_normalized_accuracy_with_pipelineID.csv')
data_set_pipelines.rename(columns = {'Unnamed: 0':'ID'}, inplace = True) 
hyperparameter_settings=pd.read_json(r'pipelines.json')


algorithms=['extra_trees', 'k_nearest', 'xgradient_boosting', 'quadratic_discriminant_analysis', 'bernoulli', 'gradient_boosting', 'decision_tree', 'multinomial', 'lda', 'random_forest']
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
       'libsvm_svc_tol', 'libsvm_svc_shrinking', 'libsvm_svc_gamma'],'quadratic_discriminant_analysis':['qda_reg_param'],'bernoulli':['bernoulli_nb_alpha',
       'bernoulli_nb_fit_prior'],'gradient_boosting':['gradient_boosting_loss',
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
       'random_forest_max_depth']}     


dataset_train=[10, 1020, 1459, 1460, 1464, 1467, 1471, 1472, 1473, 1475, 1481, 1482, 1021, 1486, 1487, 1488, 1489, 1496, 1498,
                      1500, 1501, 1507, 1508, 1022, 151, 1510, 1512, 1513, 1516, 1517, 1518, 1519, 1520, 1527, 1025, 1528, 1529, 1530,1531,
                      1532, 1533, 1534, 1535, 1536, 1537, 1026, 1538, 1539, 1540, 1541, 1546, 1556, 1557, 1561, 1563, 1564, 1036 1565,1568,
                      16, 1600, 164, 18, 180, 181, 183, 1038, 184, 187, 189, 197, 20, 209, 21, 22, 223, 225, 1040, 227, 23, 230, 26, 275,
                      276, 277, 278, 279,1467, 1471, 1472, 1473, 1475, 1481,1482, 1021, 1486, 1487, 1488, 1489, 1496, 1498,1500, 1501,
                      1507, 1508,1022, 151, 1510, 1512, 1513, 1516, 1517, 1518, 1519, 1520, 1527, 1025, 1528, 1529, 1530, 1531]

remaining_datasets=[1532, 1533, 1534, 1535, 1536, 1537, 1026, 1538, 1539, 1540, 1541, 1546, 1556, 1557, 1561, 1563, 1564, 1036, 1565, 1568,
 16, 1600, 164, 18, 180, 181, 182, 183, 1038, 184, 187, 189, 197, 20, 209, 21, 22, 223, 225, 1040, 227, 23, 230, 26, 275,
 276, 277, 278, 279, 28, 1041, 285, 287, 292, 294, 298, 3, 30, 300, 307, 31, 1043, 310, 312, 313, 32, 329, 334, 335, 336,
 337, 338, 1005, 1044, 339, 343, 346, 36, 37, 375, 377, 383, 384, 385, 1046, 386, 387, 388, 389, 39, 391, 392, 394, 395,
 397, 1048, 398, 40, 400, 401, 40474, 40475, 40476, 40477, 40478, 41, 1049, 4134, 4135, 4153, 43, 4340, 44, 444, 446, 448,
 450, 1050, 4534, 4538, 457, 458, 459, 46, 463, 464, 465, 467, 1054, 468, 469, 472, 475, 476, 477, 478, 479, 48, 480, 1055,
 50, 53, 54, 59, 6, 60, 62, 679, 682, 683, 1056, 685, 694, 713, 714, 715, 716, 717, 718, 719, 720, 1059, 721, 722, 723, 724,
 725, 726, 727, 728, 729, 730, 1060, 731, 732, 733, 734, 735, 736, 737, 740, 741, 742, 1006, 1062, 743, 744, 745, 746, 747, 748,
 749, 750, 751, 752, 1063, 753, 754, 755, 756, 759, 761, 762, 763, 764, 765, 1065, 766, 767, 768, 769, 770, 771, 772, 773,
 774, 775, 1066, 776, 777, 778, 779, 780, 782, 783, 784, 785, 787, 1067, 788, 789, 791, 792, 793, 794, 795, 796, 797, 799,
 1068, 8, 801, 803, 804, 805, 806, 807, 808, 811, 812, 1071, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 1073, 823,
 824, 825, 826, 827, 828, 829, 830, 832, 833, 1077, 834, 835, 836, 837, 838, 841, 843, 845, 846, 847, 1078, 848, 849, 850,
 851, 853, 855, 857, 859, 860, 862, 1009, 1079, 863, 864, 866, 867, 868, 869, 870, 871, 872, 873, 1080, 874, 875, 876, 877,
  878, 879, 880, 881, 882, 884, 1081, 885, 886, 887, 888, 889, 890, 891, 893, 895, 896, 1082, 900, 901, 902, 903, 904, 905, 906, 907, 908,
  909,1084, 910, 911, 912, 913, 915, 916, 917, 918, 919, 920, 11, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 1100,
 932, 933, 934, 935, 936, 937, 938, 941, 942, 943, 1106, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 1107, 955, 956,
 958, 959, 962, 970, 973, 974, 976, 977, 1115, 978, 979, 980, 983, 987, 988, 991, 994, 995, 996, 1012, 1116, 997, 1117, 1120,
 1121, 1123, 1124, 1125, 1126, 1127, 1132, 1014, 1133, 1135,1136, 1137, 1140, 1141, 1143, 1145, 1147, 1148, 1015, 1149, 1150,
 1151, 1152, 1153, 1156, 1157, 1158, 1159, 1160, 1016, 1162, 1163, 1164, 1165, 1167, 12, 14, 1412, 1442, 1443, 1019, 1444, 1446,
 1447, 1448, 1449, 1450, 1451, 1453, 1454, 1457]


def select_random_datasets(i,remaining_datasets):
    '''
    this function collects random datasets for testing. i stand for the number and remaining dataset for the
    list where to choose from
    '''
    random_datasets_for_testing=random.sample(remaining_datasets, i)
    return random_datasets_for_testing

random_10=select_random_datasets(4,remaining_datasets)


min_max_scaler = preprocessing.MinMaxScaler()#this is the preprocessing for rescaling of accuracy  
temp_list=dataset_train + random_10 #merging of the test and train dataset for rescaling
for element in temp_list:
    data_set_pipelines[[str(element)]] = min_max_scaler.fit_transform(data_set_pipelines[[str(element)]]) #actual rescaling


def quantile_treshold(start,quantile_value,dataset):
    '''
    This function calculates the threshold value based on quantiles. Start stands for the overall dataset with pipelines. 
    Datasets is the dataset where the treshold needs to be cacluated for and quantile_value is the given quantile.
    '''
    clean_score = start[str(dataset)]
    threshold=np.quantile(clean_score, quantile_value) 
    return threshold

def dataset_non_nan(dataset):
    '''
    This function filters on NAN and other unvalid values in the dataset. 
    The dataset input is the dataset that needs to be cleaned.
    '''
    pre_clean=data_set_pipelines.dropna(subset=[str(dataset)])
    return pre_clean

def transform_function(dataset):
    '''
    This is a helper function that transforms the given input dataset on ID to allow for usage within the model.
    The input dataset is a dataset you would like to change.
    '''
    dictionary_good_bad={}
    dictionary_good_bad['selected']=dataset['ID']
    return dictionary_good_bad

def split_good_bad(threshold, dataset,dataset_scores):
    '''
    This function helps to split the dataset in "good" and "bad".
    The input is the threshold and the dataset containing scores.
    '''
    dictionary_good_bad={}
    good=dataset_scores.loc[dataset_scores[dataset] >= threshold]
    bad=dataset_scores.loc[dataset_scores[dataset] < threshold]
    dictionary_good_bad['good']=good['ID']
    dictionary_good_bad['bad']=bad['ID']
    return dictionary_good_bad

def create_alogirthm_based_dataset(dataset,algorithm, dictionary):
    '''
    This functions is used to separate the pipeline configurations based on the algorithms they represent.
    Algorithm stands for the algorithm you want to separate and dictionary=dic, containing algortihm hp names. 
    Dataset represents the number of the datasets you would like to use.
    '''
    columns=dictionary['pre_processing'] + dictionary[algorithm]+dictionary['id']
    usable_dataset=dataset[columns]
    for element in dictionary[algorithm]:
        total_clean=usable_dataset[usable_dataset[element].notna()]
    return total_clean

def building_kde(dataset):
    '''
    Function for building kde, which uses the cleaned dataset and the dictionary on the algorithm and the 
    key is the algorithm name in str.
    '''
    space=dataset.to_numpy()
    count=space.shape[1]
    dens_u_kde = sm.nonparametric.KDEMultivariate(data=space,var_type=count*'o', bw='normal_reference')#weer terug verandere o
    return dens_u_kde

def clean_test_function(dataset,dictionary):
    '''
    This is the overall data cleaning function, which removes and cleans all the different pipeline configurations.
    Dataset= is the dataset you are analyzing and want to clean, and dictionary is the algorithm dictionary
    '''
    record={}
    new_dataset=dataset.loc[:, dataset.columns != 'id']
    objects=list(new_dataset.select_dtypes(include="object").columns)
    new_dataset.fillna(-1, inplace=True)
    for element in objects:
        names=new_dataset[element].unique()
        mapnames=dict(zip(names,range(len(names))))
        for elements in list(mapnames.keys()):
            if elements == None:
                mapnames[None]=-1
        record[element]=mapnames
        new_dataset[element]=new_dataset[element].map(mapnames)
        new_dataset[element] = new_dataset[element].astype(float)
    result=new_dataset
    result['id']=dataset['id']
    return new_dataset,record

def build_framework(datasets,data_set_pipelines,dictionary,algorithm):
     '''
     This function creates the framework of kde that will be used. The dataset_list goes in 
    the algorithm dicationary and the alogrihm that you want to test.
    '''
    fault_list=[]
    frame_work={}
    f = IntProgress(min=0, max=len(datasets))
    display(f)
    for dataset in datasets:
        start=dataset_non_nan(dataset)
        threshold=quantile_treshold(start,0.85,dataset)
        pipelines=transform_function(start)
        pipeline_hp=hyperparameter_settings[hyperparameter_settings.id.isin(pipelines['selected'])]
        algorithm_based_datset=create_alogirthm_based_dataset(pipeline_hp,algorithm, algarithm_dictionary)
        filled_missing=clean_test_function(algorithm_based_datset,dictionary)[0]
        good_bad_dictionary=split_good_bad(threshold, str(dataset),start)
        good=filled_missing[filled_missing.id.isin(good_bad_dictionary['good'])]
        bad=filled_missing[filled_missing.id.isin(good_bad_dictionary['bad'])]
        f.value += 1
        try:
            good_kde=building_kde(good.loc[:, good.columns != 'id'])
            bad_kde=building_kde(bad.loc[:, bad.columns != 'id'])
            weight=1/len(datasets)
            frame_work[dataset]={'good kde':good_kde, 'bad kde': bad_kde, 'weight':weight }
        except: fault_list.append(dataset)
    return frame_work,fault_list


def pre_pair_test_data(dataset,algorithm,algarithm_dictionary):
    '''
    This function prepares the test data to reshape and clean it.
    '''
    start=dataset_non_nan(dataset)
    pipelines=transform_function(start)
    pipeline_hp=hyperparameter_settings[hyperparameter_settings.id.isin(pipelines['selected'])]
    algorithm_based_datset=create_alogirthm_based_dataset(pipeline_hp,algorithm, algarithm_dictionary)
    clean=clean_test_function(algorithm_based_datset,algarithm_dictionary)
    return clean[0]

def start_collecting_random_pipelines(dataset,algorithm,algarithm_dictionary,number):
    '''
    This is the starter function to collect random pipelines for testing purposes. 
    ''' 
    data_set_new=pre_pair_test_data(dataset,algorithm,algarithm_dictionary)
    new=data_set_new.sample(n = number)
    return new


def ei_calculation(random_pipelines,algorithm,dictionary,framework):
    '''
    This function calculates the expected improment for all the random pipeline configurations.
    '''
    final_function=clean_test_function(random_pipelines,dictionary)[0]
    rows=final_function.loc[:, random_pipelines.columns != 'id'].to_dict('index')
    frame_dic={}
    for ds in framework.keys():
        empt_list=[]
        for element in list(rows.keys()):
            prediction_kde_good=framework[ds]['good kde'].pdf(data_predict=np.array(list(rows[element].values())))
            prediction_kde_bad=framework[ds]['bad kde'].pdf(data_predict=np.array(list(rows[element].values())))
            ei=prediction_kde_good/prediction_kde_bad
            empt_list.append((element,ei))
        new=sorted(empt_list, key = lambda x: x[1])
        frame_dic[ds]=new
    return frame_dic


def tau_weight_function(prediction, data_set_pipelines,random_pipelines,test_dataset):
    '''
    This function calculates Kendall’s tau ranking correlation, based on the real ranking and the predicted ranking.
    '''
    overall_dictionary={}
    order=good_order(random_pipelines,str(test_dataset),data_set_pipelines)[1]
    for db in list(prediction.keys()): 
        predicted=[element[0] for element in prediction[db]]
        predict_list=[]
        correct_list=[]
        for element in order:
            postion_predict=predicted.index(element)
            postion_correct=order.index(element)
            correct_list.append(postion_correct)
            predict_list.append(postion_predict)
        tau, p_value = stats.kendalltau(correct_list, predict_list)
        overall_dictionary[db]=tau
    
    return overall_dictionary

def reweighting(weights,framework):
    '''
    This function reweights the mtm based on the tau score. 
    '''
    new_weights={}
    for db in list(framework.keys()):
        if weights[db]>0:
            new_weights_value=weights[db]
        if weights[db]<=0:
            new_weights_value=0
        new_weights[db]=new_weights_value
    total=sum(new_weights.values())
    for key in list(new_weights.keys()):
        try:
            new=new_weights[key]/total
        except: new=0
        old=framework[key]['weight']
        framework[key]['weight']=0
    return framework

def good_order(random_piplines,dataset,data_set_pipelines):
    '''
    This function orders the pipelines based on their performance.
    '''
    pipelines=data_set_pipelines.loc[data_set_pipelines['ID'].isin(list(random_piplines))]
    sorted_right_order=pipelines.sort_values(by=[dataset])
    ordered=list(sorted_right_order.index)
    result=list(sorted_right_order['ID'])[::-1]
    return result, ordered,sorted_right_order


def basic_model(data_sets,data_set_pipelines,algarithm_dictionary,hyperparameter_settings,test_dataset,nr):
    '''
    This function builds the initial mtm. The function needs all the data including the pipeline configurations and 
    the dictionary with the different algorithms and the test data. 
    ''' 
    count=0
    dic_basic_models={}
    for element in algorithms:
        print(element)
        frame_work=build_framework(dataset_train,data_set_pipelines,algarithm_dictionary,element)
        random_pipelines=start_collecting_random_pipelines(test_dataset,element,algarithm_dictionary,10) 
        ei=ei_calculation(random_pipelines,element,algarithm_dictionary,frame_work[0] )
        weights=tau_weight_function(ei, data_set_pipelines,random_pipelines['id'],test_dataset)
        weighed_framework=reweighting(weights,frame_work[0])
        dic_basic_models[element]=weighed_framework
    return dic_basic_models



def ei_calculation_algorithm_kde(random_pipelins,algorithm,dictionary,framework):
    '''
    This function calculates the expected improvement for both the good and the bad kde's based on the random piplines
    the build the mtm and the algorithms.
    '''
    hyper_parameters=random_pipelins.loc[:, random_pipelins.columns != 'id']
    rows=hyper_parameters.to_dict('index')
    frame_dic={}
    store_dic={}
    for element in list(rows.keys()):
        empt_list=[]
        temp_list=[]
        for ds in framework.keys():
            kde_good=framework[ds]['good kde'].pdf(data_predict=np.array(list(rows[element].values())))
            kde_bad=framework[ds]['bad kde'].pdf(data_predict=np.array(list(rows[element].values())))
            ei=kde_good/kde_bad
            weight=framework[ds]['weight']
            if str(ei)== '-inf' or str(ei)== 'nan' or str(ei)== 'inf':
                weighted=(0.1*10**-100)*weight
                empt_list.append(weight)
            else:
                weighted=ei*weight
                empt_list.append(weighted)
            #print(element)    
            temp_list.append((ds,weighted))
        store_dic[element]=temp_list
        
        new=sum(empt_list)
        frame_dic[element]=new
    sorted_d = sorted(frame_dic.items(), key=lambda x: x[1])
    final=[element[0] for element in sorted_d]
    return store_dic, final[-1:],sorted_d[-1:]

def change_function(input_values): 
    '''
    This function changes the format to make it easier to check.
    '''
    nieuwdict = {}
    final_dict= {}
    for i in input_values.keys():
        lijst = input_values.get(i)
        for j in lijst:
            tupel = j
            instance = nieuwdict.get(tupel[0], None)
            if instance == None:
                nieuwdict.update({tupel[0]: [(i,tupel[1])]})
            else:
                instance.append((i,tupel[1]))
    for element in nieuwdict.keys():
        new=sorted(nieuwdict[element], key = lambda x: x[1])
        final_dict[element]=new
    return final_dict


def iterations(nr,dic_basic_models,algarithm_dictionary,test_dataset):
    '''
    This function is used to execute the search of the mtm model multiple times while learning from previous experiences.
    The input is the nr of iterations the basic models built, the alogrithm dictionary and the testdataset.
    '''
    dic_predictions={}
    overal_dic={}
    overall_random_pipelines={}
    count=0
    f = IntProgress(min=0, max=nr)
    display(f)
    while count < nr:
        store_random_pipelines={}
        for element in list(dic_basic_models.keys()):
            random_piplines_model=start_collecting_random_pipelines(test_dataset,element,algarithm_dictionary,100) 
            prediction=ei_calculation_algorithm_kde(random_piplines_model,element,algarithm_dictionary,dic_basic_models[element])        
            store_random_pipelines[str(element)]=random_piplines_model['id']
            dic_predictions[element]=prediction[2]
            ei=change_function(prediction[0])
            weights=tau_weight_function(ei, data_set_pipelines,random_piplines_model['id'],test_dataset)
            weighed_framework=reweighting(weights,dic_basic_models[element])
        overall_random_pipelines[str(count)]=store_random_pipelines
        overal_dic[str(count)]=dic_predictions
        dic_predictions={}
        count+=1
        f.value += 1
    return overal_dic,weighed_framework,overall_random_pipelines

def runner_function(test_sets):
    '''
    This is the runner function which builds the basic mtm and then tests it on the random datasets and
    lets the model learn from previous experiences.
    '''
    count=0
    temp_dic={}
    prediction_dictionary_random_search={}
    for elements in test_sets:
        try:
            print(elements)
            round_dictionary=basic_model(dataset_train,data_set_pipelines,algarithm_dictionary,hyperparameter_settings,elements,10)
            predictions=iterations(10,round_dictionary,algarithm_dictionary,elements)
            prediction_dictionary_random_search[elements]=predictions[2]
            total={}
            temp_list22=[]
            for troep in list(predictions[0]['0'].keys()):
                temp_list22=[]
                for element in list(predictions[0].keys()):
                    value=predictions[0][element][troep][0][0]
                    accurency=data_set_pipelines[str(elements)].loc[value]
                    temp_list22.append(accurency)
                total[troep]=temp_list22
            temp_dic[elements]=total
        except:
            count+=1
            continue
    a_file = open("data.pkl", "wb")
    pickle.dump(prediction_dictionary_random_search, a_file)
    a_file.close()
    return temp_dic,count

results=runner_function(random_10)


def take_average_for_outcome(algorithms,results):
    '''
    This function takes the recommended pipeline configurations for each algorithm of the different iterations and determines the 
    the average score per algorithm per iteration.
    '''
    dic_overall={}
    for algorithm in algorithms:
        score_list=[]
        for element in list(range(10)):
            empty_list=[]
            for keys in results[0].keys():
                empty_list.append(results[0][keys][algorithm][element])
            average=sum(empty_list)/(len(empty_list)-results[1])
            score_list.append(average)
        dic_overall[algorithm]=score_list
    return dic_overall
dic_results=take_average_for_outcome(algorithms,results)

a_file = open("random_search_results.pkl", "rb") #this function opens the results of the random search 
Results_random_search = pickle.load(a_file)


def visualisation(alogirhtm):
    '''
    This is the function that visualizes the outcomes of both the random search and the mtm. 
    '''
    temp_dic={}
    temp_dic['Random_search']=Results_random_search[alogirhtm]
    temp_dic['MTM']=dic_results[alogirhtm]
    for k, v in temp_dic.items():
        axes = plt.gca()
        axes.set_xlim([0,11
                      ])
        axes.set_ylim([0.0,1.1])
        plt.plot(range(1, len(v) + 1), v, '.-', label=k)
        plt.legend()
        plt.xlabel("Number of iterations")
        plt.title(str(alogirhtm))
        plt.ylabel("Accuracy")
        plt.plot()


visualisation('quadratic_discriminant_analysis')
plt.savefig('quadratic_discriminant_analysis.png')

visualisation('random_forest')
plt.savefig('random_forest.png')

visualisation('lda')
plt.savefig('lda')

visualisation('decision_tree')
plt.savefig('decision_tree')


visualisation('gradient_boosting')
plt.savefig('gradient_boosting')


visualisation('bernoulli')
plt.savefig('bernoulli')


visualisation('xgradient_boosting')
plt.savefig('xgradient_boosting')


visualisation('k_nearest')
plt.savefig('k_nearest')


visualisation('extra_trees')
plt.savefig('extra_trees')


visualisation('quadratic_discriminant_analysis')
plt.savefig('quadratic_discriminant_analysis')


visualisation('multinomial')
plt.savefig('multinomial')
