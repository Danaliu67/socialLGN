import time
from os.path import join

import torch

import Procedure
import register
import utils
import world
import pandas as pd
from register import dataset

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
torch.autograd.set_detect_anomaly(True)
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        print(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")


loss = Procedure.BPR_train_original(dataset, Recmodel, bpr, 0)

u_batch_size = world.config['test_u_batch_size']
trainDict: dict = dataset._trainDic

result = []
with torch.no_grad():
    users = list(trainDict.keys())
    GPU = torch.cuda.is_available()
    device = torch.device('cuda' if GPU else "cpu")
    for batch_users in utils.minibatch(users, batch_size=u_batch_size):
        batch_users_gpu = torch.tensor(batch_users, dtype=torch.long).to(device)
        rating = Recmodel.getUsersRating(batch_users_gpu)
        allPos = dataset.getUserPosItems(batch_users)
        _, rating_K = torch.topk(rating, k=world.rec_topk)
        for idx, user in enumerate(batch_users):
            result.extend([[user, int(item)] for item in rating_K[idx] if item not in allPos[idx]])

result_df = pd.DataFrame(result, columns=['user', 'item'])
result_df.to_csv(f'./data/preprocessed/{dataset.src}_data/train_set_predict_{dataset.idx}.txt', index=False)
