import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd

POSSIBLE_PATHS = [ "e:\\oneDrive\\UOB\\Federated_XGBoost_Python-main\\data\\", \
    "/data/BioGrid/meerhofj/Database/", \
                      "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
                      "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
dataset = 'purchase-10' 
dataset_list = ['purchase-10', 'purchase-20', 'purchase-50', 'purchase-100', 'texas', 'MNIST', 'synthetic', 'Census', 'DNA']

def apply_DP(X, y):
    import pydp
    pydp.algorithms.laplacian.BoundedStandardDeviation
    pydp.LaplacePrivacy({'epsilon':epsilon})
    sensitivity = 1
    epsilon = 1
    scale = sensitivity/epsilon
    noise = np.random.laplace(0, scale, X.shape)
    X + noise
    return X, y

def split_D(D, federated, train_size, n_shadows, fName):

    X = D[0]
    y = D[1]
    total_size = X.shape[0]

    if federated:
        assert (train_size*2) + ((train_size//2) * n_shadows) <= total_size
    else:
        assert train_size*4 <= total_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:int(train_size*1.25)]
    y_test = y[train_size:int(train_size*1.25)]
    if federated:
        begin = train_size*2
        shadow_size = train_size//2
        X_shadow = [X[i:i+shadow_size] for i in range(begin, begin+(shadow_size*n_shadows), train_size//2)]
        y_shadow = [y[i:i+shadow_size] for i in range(begin, begin+(shadow_size*n_shadows), train_size//2)]
    else:
        X_shadow = X[train_size*2:train_size*4]
        y_shadow = y[train_size*2:train_size*4]


    return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow

def check_mul_paths(filename, paths):
    import pickle
    for path in paths:
        try:
            with open(path + filename, 'rb') as file:
                obj = pickle.load(file)
                return obj
        except FileNotFoundError:
            continue
    raise FileNotFoundError("File not found in all paths :(")

def check_mul_paths_csv(filename, paths):
    for path in paths:
        # print(f"testing {path+ filename + '.csv'}")
        if os.path.exists(path+ filename + '.csv'):
            return pd.read_csv(path + filename + ".csv")
        print(f"file {path+ filename + '.csv'} not found")
    raise FileNotFoundError("File not found in all paths :(")

def take_and_remove_items(arr, size, seed=0): #sshoutout to Chat-gpt
    np.random.seed(seed)
    indices = np.random.choice(len(arr), size,replace=False )
    selected_items = np.take(arr, indices, axis=0)
    arr = np.delete(arr, indices, axis=0)
    return selected_items, arr

def makeOneHot(y):
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    encoder.fit(y)
    y = encoder.transform(y).toarray()
    return y

def getPurchase(num, paths, federated=False, train_size=20_000):
    #first try local
    # 
    # logger.warning(f"getting purchase {num} dataset!")

    n_shadows = 10
    def returnfunc():
        X = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_features.p', paths)
        y = check_mul_paths('acquire-valued-shoppers-challenge/' + 'purchase_100_' + str(num) + '_labels.p', paths)
        y = y.reshape(-1, 1)
        y = makeOneHot(y) 
        fName = []
        for i in range(600):
            fName.append(str(i))
        return split_D((X, y), federated, train_size, n_shadows, fName)        
    return returnfunc

def getTexas(paths, federated=False, train_size=20_000):
    """
    size = 925_128
    features = 11
    n_classes = 100


    Args:
        paths (_type_): _description_
        federated (bool, optional): _description_. Defaults to False.
    """
    def returnfunc():
        # logger.warning("getting Texas database!")
        n_shadows = 10

        
        X = check_mul_paths('texas/' + 'texas_100_v2_features_2006.p', paths)
        X = np.array(X)
        shape = np.shape(X) # 2516991, 11 2006 = 925128, 11
        y = check_mul_paths('texas/' + 'texas_100_v2_labels_2006.p', paths) 
        # fName = check_mul_paths('texas/' + 'texas_100_v2_feature_desc.p', paths)
        fName = ['THCIC_ID', 'SEX_CODE', 'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION', \
             'LENGTH_OF_STAY', 'PAT_AGE', 'PAT_STATUS', 'RACE', 'ETHNICITY', \
                'TOTAL_CHARGES', 'ADMITTING_DIAGNOSIS']
        X = X[100_000:]
        y = y[100_000:]
        y = y.reshape(-1, 1)
        # print(np.unique(y[:train_size]))
        y = makeOneHot(y)
        
        return split_D((X,y), federated, train_size, n_shadows, fName)
    return returnfunc



# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getTexas(POSSIBLE_PATHS)()


def getMNIST(paths, federated=False, train_size=2_000):
    """only of size 1797!

    Args:
        paths (_type_): _description_
        federated (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    from sklearn import datasets

    train_size = 2000
    n_shadows = 10
    
    def returnfunct():
        digits = datasets.load_digits()
        images = digits.images
        targets = digits.target
        images = images.reshape(1797, 8*8)
        fName = digits.feature_names
        X = images
        y = targets
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        return split_D((X, y), federated, train_size, n_shadows, fName)
    return returnfunct  
    
# getMNIST(None, False)()

def getSynthetic(federated=False, n_classes=8, train_size=2_000):
    random_state = 420
    shadow_size = train_size*2   
    n_shadows = 10
    n_features = 16
    if federated:
        shadow_size = train_size //2
    else:
        shadow_size = train_size

    def returnfunc():
        """returns ndarrays with all X types and y where y is One Hot encoded

        Returns:
            _type_: _description_
        """
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=(train_size*2)+(shadow_size * n_shadows),
                                    n_features=n_features, n_informative=8, n_redundant=0, n_clusters_per_class=1, 
                                    class_sep=1.0, n_classes=n_classes, random_state=random_state)
        y = y.reshape(-1, 1)
        y = makeOneHot(y)
        fName = [str(i) for i in range(0, n_features)]

        return split_D((X,y), federated, train_size, n_shadows, fName)
    return returnfunc
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getSynthetic(False, 10, 50_00)()
# x = 1
def getCensus(paths, federated=False, train_size=2000):  # binary issue
    pass

def getDNA(paths, federated=False, train_size=2000):
    return

def getWine(federated=False, train_size=2000):
    train_size = 10_000
    n_shadows = 10

    def returnfunc():
        from ucimlrepo import fetch_ucirepo 
        # fetch dataset 
        wine_quality = fetch_ucirepo(id=186) 
        
        # data (as pandas dataframes) 
        X = wine_quality.data.features 
        y = wine_quality.data.targets
        fName = wine_quality.data.features.columns
        X = wine_quality.data.features.values   
        y = wine_quality.data.targets.values
        y = y[:,0]

        from imblearn.over_sampling import SMOTE
        x_new, y_new = SMOTE(sampling_strategy='auto', random_state=666, k_neighbors=4).fit_resample(X, y)
        X = np.vstack((X, x_new))
        y = np.hstack((y, y_new))
        return split_D((X, y), federated, train_size, n_shadows, fName)
    return returnfunc
# getWine(False)()

def getHealthcare_split(paths):
    def returnfunc():
        train = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data", paths)
        non_continuous = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age', 'Stay']
        train = train.dropna()

        for featureName in non_continuous:
            train[featureName] = train[featureName].factorize()[0]  # String to int

        fName = train.columns.tolist()[1:17]
        X = train.values[:, 1:17]
        y = makeOneHot(y = train.values[:, 17].reshape(-1,1))
        # Xtmp = np.ndarray.tolist(X)
        data_mask_user1 = (X[:, 0] == 23)
        data_mask_user2 = (X[:, 0] == 26)
        data_user1 = X[data_mask_user1]  # 26566
        data_user2 = X[data_mask_user2]  # 33076
        y_user1 = y[data_mask_user1]
        y_user2 = y[data_mask_user2]
        datadevision_hardcode = [1000, 4000]
        assert (datadevision_hardcode[0] * 4) < 26566 and (datadevision_hardcode[1] * 4) < 33076 
        X_train = np.vstack( (data_user1[ :datadevision_hardcode[0], :], data_user2[ :datadevision_hardcode[1], :]) )
        y_train = np.vstack( (y_user1[ :datadevision_hardcode[0]], y_user2[ :datadevision_hardcode[1]]) )
        
        X_test = np.vstack( (data_user1[datadevision_hardcode[0]: datadevision_hardcode[0]*2, :], data_user2[ datadevision_hardcode[1]:datadevision_hardcode[1]*2, :]) )
        y_test = np.vstack( (y_user1[datadevision_hardcode[0]: datadevision_hardcode[0]*2 ], y_user2[datadevision_hardcode[1]:datadevision_hardcode[1]*2]) )

        X_shadow = np.vstack( (data_user1[datadevision_hardcode[0]*2: datadevision_hardcode[0]*3, :], data_user2[ datadevision_hardcode[1]*2:datadevision_hardcode[1]*3, :],
                               data_user1[datadevision_hardcode[0]*3: datadevision_hardcode[0]*4, :], data_user2[ datadevision_hardcode[1]*3:datadevision_hardcode[1]*4, :]) )

        y_shadow = np.vstack( (y_user1[datadevision_hardcode[0]*2: datadevision_hardcode[0]*3 ], y_user2[datadevision_hardcode[1]*2:datadevision_hardcode[1]*3],
                               y_user1[datadevision_hardcode[0]*3: datadevision_hardcode[0]*4 ], y_user2[datadevision_hardcode[1]*3:datadevision_hardcode[1]*4]) )

        return X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow
    return returnfunc
X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getHealthcare_split(POSSIBLE_PATHS)()
x = 1

def getHealthcare(paths, federated=False, train_size=2_000): # https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii
    """retireves the Healtcare dataset from kaggle https://www.kaggle.com/datasets/nehaprabhavalkar/av-healthcare-analytics-ii
    general information:
    nFeatures = 16 
    nClasses = 11
    nUsers = ~318k
    I'm not using test_data.csv and sample_sub.cvs as there it is only testing the classification of 0-10 days y/n?

    for a federated data retrieval 
    Args:
        paths (_type_): _description_
        federated (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    n_shadows = 300
    # A MINIMUM OF 4 SHADOWS ARE NEEDED!!!
    def returnfunc():
        train = check_mul_paths_csv("AV_HealthcareAnalyticsII\\train_data", paths)
        # train_dp = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data_dp", paths)

        # x = pd.read_csv('/data/BioGrid/meerhofj/Database/AV_HealthcareAnalyticsII/train_data_dp.tsv', sep='\t')

        # test = check_mul_paths_csv("AV_HealthcareAnalyticsII/test_data", paths)
        # dict = check_mul_paths_csv("AV_HealthcareAnalyticsII/train_data_dictionary", paths)
        # sample = check_mul_paths_csv("AV_HealthcareAnalyticsII/sample_sub", paths)
        non_continuous = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age', 'Stay']
        train = train.dropna()

        for featureName in non_continuous:
            train[featureName] = train[featureName].factorize()[0]  # string to int. categorical encoding

        # train[strings] = train[strings].apply(lambda x: pd.factorize(x)[0])
        # train = train.apply(lambda x: pd.factorize(x)[0])
        
        fName = train.columns.tolist()[1:17]
        X = train.values[:, 1:17]
        y = makeOneHot(y = train.values[:, 17].reshape(-1,1))

        return split_D((X,y), federated, train_size, n_shadows, fName)

    # data = np.genfromtxt(paths + "AV_HealthcareAnalyticsII/train_data.csv")
    return returnfunc


# POSSIBLE_PATHS = ["/data/BioGrid/meerhofj/Database/", \
#                       "/home/hacker/jaap_cloud/SchoolCloud/Master Thesis/Database/", \
#                       "/home/jaap/Documents/JaapCloud/SchoolCloud/Master Thesis/Database/"]
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getHealthcare(POSSIBLE_PATHS, True)()
# n_shadows = len(X_shadow)
# x=1
def getDataBase(dataBaseName, paths, federated=False, train_size=2000):
    """After setting the database in the config, this will retrieve the database
    """
    get_databasefunc = {'purchase-10': getPurchase(10, paths, federated, train_size), 'purchase-20':getPurchase(20, paths, federated, train_size), 
                    'purchase-50':getPurchase(50, paths, federated, train_size), 'purchase-100':getPurchase(100, paths, federated, train_size), 
                    'texas':getTexas(paths, federated, train_size), 'healthcare':getHealthcare(paths, federated, train_size), 'MNIST':getMNIST(paths, federated, train_size), 
                    'synthetic-10':getSynthetic(federated, 10, train_size), 'synthetic-20':getSynthetic(federated, 20, train_size), 
                    'synthetic-50':getSynthetic(federated, 50, train_size), 'synthetic-100':getSynthetic(federated, 100, train_size), 
                    'Census':getCensus(paths, federated, train_size), 'DNA':getDNA(paths, federated, train_size),
                    'healthcare_hardcoded':getHealthcare_split(paths)
                   }[dataBaseName]
    return get_databasefunc

def getConfigParams(dataBaseName): # retreive n_classes, n_features
    """shamefully hardcoded nClasses and nFeatures retriever. 
    

    Args:
        dataBaseName (str): name of the datset

    Returns:
        tuple(int, int): tuple of (nClasses, nFeatures)
    """
    get_databasefunc = {'purchase-10': (10, 600), # nClasses, nFeatures
                        'purchase-20': (20, 600), 
                        'purchase-50': (50, 600), 
                        'purchase-100': (100, 600), 
                        'synthetic-10': (10, 16), # nClasses, nFeatures
                        'synthetic-20': (20, 16), 
                        'synthetic-50': (50, 16), 
                        'synthetic-100': (100, 16), 
                        'texas':(100, 11), 
                        'healthcare':(11, 16),
                        'healthcare_hardcoded':(11, 16),
                        'MNIST':(10, 64), 
                        'Census':(-1, -1), 
                        'DNA':(-1, -1),
                        'iris':(3, 4),
                        'credit_card_fraud': (2, 30),
                        'whales_prediction': (2, 58)
                   }[dataBaseName]
    return get_databasefunc[0], get_databasefunc[1]

# x = 1