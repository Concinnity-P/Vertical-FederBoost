from SFXGBoost.Model import SFXGBoost, devide_D_Train, VFLXGBoost
from SFXGBoost.config import Config, MyLogger
from SFXGBoost.dataset.datasetRetrieval import getDataBase
from ddsketch import DDSketch
# from SFXGBoost.data_structure.databasestructure import QuantiledDataBase,  DataBase
import numpy as np
from SFXGBoost.view.plotter import plot_loss
import pickle
import os
from copy import deepcopy
import pandas as pd
from datetime import date
import time
from phe import paillier
public_key, private_key = paillier.generate_paillier_keypair(n_length=100)
from typing import List
from SFXGBoost.data_structure.treestructure import SplittingInfo
from SFXGBoost.loss.softmax import getLoss

dataset = 'whales_prediction' #'whales_prediction'# 'iris' #'healthcare'
NUM_CLIENTS = 3
DATA_DEVISION = [1/NUM_CLIENTS] * NUM_CLIENTS

NUM_FEATURE = 58
NUM_FEATURE_CLIENT = [18, 20, 20]

config = Config(experimentName = "experiment 1",
        nameTest= dataset + " test",
        model="normal",
        dataset=dataset,
        lam=1, # 0.1 10
        gamma=0, # 0.5
        alpha=0.0,
        learning_rate=0.3, #0.3, 0.75
        max_depth=3,
        max_tree=10,
        nBuckets=100,#100
        save=False,
        data_devision=DATA_DEVISION,
        train_size=10_000,
        client=0,
        num_client=NUM_CLIENTS,
        total_feature=NUM_FEATURE,
        client_feature=0
        )

configs:list[Config] = []
for i in range(config.num_client):
    configs.append(Config(experimentName = "experiment 1",
        nameTest= config.nameTest,
        model=config.model,
        dataset=dataset,
        lam=config.lam , # 0.1 10
        gamma=config.gamma, # 0.5
        alpha=config.alpha,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_tree=config.max_tree,
        nBuckets=config.nBuckets,
        save=False,
        data_devision=config.data_devision,
        train_size=config.train_size,
        client=i, # 0 is server | 0 is the participant holds the label
        num_client=config.num_client,
        total_feature=config.total_feature,
        client_feature=NUM_FEATURE_CLIENT[i]
        ))
                          

loggers = []
for i in range(config.num_client):
    loggers.append(MyLogger(configs[i]).logger)


# POSSIBLE_PATHS = ["e:\\oneDrive\\UOB\\Federated_XGBoost_Python-main\\data\\"]
# X_train, y_train, X_test, y_test, fName, X_shadow, y_shadow = getDataBase(config.dataset, POSSIBLE_PATHS, False, config.train_size)()
#ndarray

# from sklearn import datasets

# iris = datasets.load_iris()

# X_train = iris.data

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

# y_train = encoder.fit_transform(iris.target.reshape(-1, 1)).toarray()

# fName = iris.feature_names

# data = pd.read_csv(r'E:\OneDrive\UOB\Dataset\Credit-Card-Data\Data\Credit Card Fraud Detection.csv')\
#     .groupby('Class', group_keys=False).apply(lambda x: x.sample(frac=0.1, replace=False, random_state=1))\
#     .sample(frac=1, random_state=1, replace=False)

# X_train = data.drop('Class', axis=1).values
# y_train = encoder.fit_transform(data['Class'].values.reshape(-1, 1)).toarray()
# fName = data.drop('Class', axis=1).columns

from getData import getWhaleDate

train_df_shuffle, y = getWhaleDate()

X_train = train_df_shuffle.values
y = encoder.fit_transform(y.values.reshape(-1, 1)).toarray()
fName = train_df_shuffle.columns


# # split data
total_users = config.num_client # participants

X_train_list = []
X_test_list = []
# y_train_list = []

# for rank in range(total_users):
#     rank += 1
#     X_train_my, y_train_my = devide_D_Train(X_train, y_train, rank, config.data_devision)
#     X_train_list.append(X_train_my)
#     y_train_list.append(y_train_my)


X_train_list.append(X_train[:12000,:20])
X_train_list.append(X_train[:12000,20:40])
X_train_list.append(X_train[:12000,40:])
X_test_list.append(X_train[12000:,:20])
X_test_list.append(X_train[12000:,20:40])
X_test_list.append(X_train[12000:,40:])


y_train = y[:12000]
y_test = y[12000:]

fName_list = [fName[:20], fName[20:40], fName[40:]]

# sketchs = [DDSketch() for _ in range(total_users)]
# sketch = DDSketch()
res_to_merge = []
bst_clients:list[SFXGBoost] = []
for i in range(total_users):
    bst_client = SFXGBoost(configs[i], loggers[i])
    if i == 0:
        res = bst_client.participant_fit(X_train_list[i], X_test_list[i], fName_list[i], fName, y_train, y_test  )
    else:
        res = bst_client.participant_fit(X_train_list[i], X_test_list[i], fName_list[i], fName)
    res_to_merge.append(res)
    bst_clients.append(bst_client)

# merge the results in server
splitCandidates = res_to_merge[0].copy()
splitCandidates.extend(res_to_merge[1])
splitCandidates.extend(res_to_merge[2])
# for i in range(len(res_to_merge[0])):
#     if all([isinstance(res_to_merge[p][i][1], np.ndarray)  for p in range(total_users)]):
#         combined_array = np.concatenate([res_to_merge[p][i][1] for p in range(total_users)], axis=0)
#         if np.unique(combined_array).shape[0] > config.nBuckets + 1:
#             sketch = DDSketch()
#             for j in range(total_users):
#                 sketch.merge(res_to_merge[j][i][0])
#             quantiles = np.array([sketch.get_quantile_value(q/config.nBuckets) for q in range(0, config.nBuckets, 1)])
#             splitCandidates.append(np.unique(np.round(quantiles,4)))
#         else:
#             splitCandidates.append(np.unique(combined_array))
#     else:
#         sketch = DDSketch()
#         for j in range(total_users):
#             sketch.merge(res_to_merge[j][i][0])
#         quantiles = np.array([sketch.get_quantile_value(q/config.nBuckets) for q in range(0, config.nBuckets, 1)])
#         splitCandidates.append(np.unique(np.round(quantiles,4)))

splitCandidates_dict = {fName[i]: splitCandidates[i] for i in range(len(splitCandidates))}
for i in range(total_users):
    bst_clients[i].setSplits(splitCandidates)

splitCandidates_dict_list = []
for i in range(total_users):
    splitCandidates_dict_list.append({fName_list[i][j]: res_to_merge[i][j] for j in range(len(res_to_merge[i]))})

# client update splits
for i in range(total_users):
    bst_clients[i].setquantiles(splitCandidates_dict_list[i])

# bst = SFXGBoost(configs[0], loggers[0]) # server model
# bst.server_fit(fName,splitCandidates)

# a = FitRes(Status(Code.OK, "OK"), Parameters([bst], ""), 0, {})

bst = bst_clients[0]


vfl_model = VFLXGBoost(bst_clients)

#define update function to comunicate with clients
def update(bst_clients:List[SFXGBoost], update_infos:List[List[SplittingInfo]], d):
    instances_list = []
    for c in range(config.nClasses):
        # if d == config.max_depth :
        #     break
        instances_list.append([])
        for n, update_info in enumerate(update_infos[c]):
            featureid = update_info.featureId
            # if update_info.isValid:
            if featureid is not None:
                if featureid < 20:
                    clientid = 0
                elif featureid < 40:
                    clientid = 1
                else:
                    clientid = 2
            else: 
                clientid = 0
            instances_list[c].append(bst_clients[clientid].split_instances(update_info,  c, n))
    return instances_list


# training

for t in range(config.max_tree):
    current_time = time.time()
    print(f"Tree {t}:")
    for d in range(config.max_depth + 1):
        print(f"Depth {d}:")
        if d == 0:
            G, H = bst.participant_boost(t, d)
            # GHs = []
            # for i in range(total_users):
            #     GH = bst_clients[i].participant_boost(t, d)
            #     GHs.append(GH)

            #encryption
            G_encrypted = []
            H_encrypted = []
            for c in range(config.nClasses):
                G_encrypted.append([public_key.encrypt(g) for g in G[c]])
                H_encrypted.append([public_key.encrypt(h) for h in H[c]])
            G_encrypted = np.array(G_encrypted)
            H_encrypted = np.array(H_encrypted)
        
        # gcn, hcn = bst.appendGradients(bst.currentNodes[0][0].instances ,G_encrypted[0], H_encrypted[0])
        #sending to all clients
        res = []
        for i in range(total_users):
            res.append(bst_clients[i].participant_appendGradients(G_encrypted, H_encrypted, t, d, public_key.encrypt(0)))
        
        GH =  deepcopy(res[0])

 

        for i in range(1,total_users):
            for c in range(config.nClasses):
                for n in range(len(GH[0][c])):
                    GH[0][c][n].extend(res[i][0][c][n])
                    GH[1][c][n].extend(res[i][1][c][n])

        # decrytion
        for c in range(config.nClasses):
            for n in range(len(GH[0][c])):
                for k in range(config.nFeatures):
                    GH[0][c][n][k] = [private_key.decrypt(g) for g in GH[0][c][n][k]]
                    GH[1][c][n][k] = [private_key.decrypt(h) for h in GH[1][c][n][k]]


        update_info = bst.server_boost([GH], t, d)
        # train_losses = np.array([])
        # test_losses = np.array([])
        instances_list = update(bst_clients, update_info, d)
        for i in range(total_users):
            bst_clients[i].tree_update(update_info, instances_list,t, d)
            # loss = bst_clients[i].participant_update(update_info, t, d) #return train loss and test loss
            # train_losses = np.append(train_losses, loss[0])
            # test_losses = np.append(test_losses, loss[1])
        if d == config.max_depth :
            vfl_model.update()
            train_loss = bst.losslog_train[-1]
            print(f"Train loss: {train_loss}")

            y_pred_test = vfl_model.predict_proba(None, t = t)
            y_true = bst_clients[0].y_test
            test_loss = getLoss(y_true, y_pred_test)
            print(f"Test loss: {test_loss}")
            bst.losslog_test.append(test_loss)


            # bst.losslog_train.append(np.mean(train_losses))
            # bst.losslog_test.append(np.mean(test_losses))
            if t == config.max_tree - 1:
                plot_loss(bst.losslog_train,bst.losslog_test,config)
    print(f"Time for tree {t}: {round(time.time() - current_time, 0)}s")
        
        # aggregate the loss from all the clients, log the loss


# bst_save = deepcopy(bst_clients[0])
# bst_save.X_train = None
# bst_save.y_train = None
# bst_save.X_test = None
# bst_save.y_test = None
# bst_save.original_data = None
# bst_save.logger = None




day = date.today().strftime("%b-%d-%Y")

# curTime = round(time.time())
curTime = time.strftime("%H:%M", time.localtime())

for i in range(total_users):

    
    bst_save = bst_clients[i]

    # bst_save.trees = bst_clients[i].trees
    # bst_save.fName = fName
    # bst_save.X_test = bst_clients[i].X_test
    # bst_save.y_test = bst_clients[i].y_test
    # bst_save.X_train = bst_clients[i].X_train
    # bst_save.y_train = bst_clients[i].y_train

    # print(bst_save.predict(X_train))


    os.makedirs(f'./Saves/{day}', exist_ok=True)
    pickle.dump(bst_save, open(f"./Saves/{day}/model_{curTime.replace(':','_')}_{configs[0].dataset}_{i+1}.pkl", 'wb'))

# bst_save:SFXGBoost = pickle.load(open("./Saves/model.pkl", 'rb'))
# print(bst_save.predict(X_train))
# print(bst)

