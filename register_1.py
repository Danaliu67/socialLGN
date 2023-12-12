from pprint import pprint
import time

import PreProcessedDatasetLoader
import model
# import world 


if world.dataset in ['lastfm', 'ciao', 'epinions', 'douban', 'gowalla']:
    dataset = PreProcessedDatasetLoader.SocialGraphDataset(world.dataset)

    # if world.model_name in ['SocialLGN']:
    #     dataset = PreProcessedDatasetLoader.SocialGraphDataset(world.dataset)
    # elif world.model_name in ['LightGCN']:
    #     dataset = PreProcessedDatasetLoader.GraphDataset(world.dataset)
    # elif world.model_name in ['bpr']:
    #     dataset = PreProcessedDatasetLoader.PairDataset(world.dataset)

# print('===========config================')
# pprint(world.config)
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# print("LOAD:", world.LOAD)
# print("Weight path:", world.PATH)
# print("Test Topks:", world.topks)
# print("using bpr loss")
# print('===========end===================')

MODELS = {
    'bpr': model.PureBPR,
    'LightGCN': model.LightGCN,
    'SocialLGN': model.SocialLGN,
}