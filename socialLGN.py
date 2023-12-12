import os
from os.path import join
import torch

import PreProcessedDatasetLoader
from model import SocialLGN

class SocialLGNModel:
    def __init__(self):
        self.config = {
            'layer': 3,
            'bpr_batch_size': 2048,
            'latent_dim_rec': 64,
            'lr': 0.001,
            'decay': 1e-4,
            'test_u_batch_size': 100,
        }
    
    

    def topk_recommendation(self, graph, extra_parameter):
        user_id, k = extra_parameter

        dataset = PreProcessedDatasetLoader.SocialGraphDataset(graph)

        model = SocialLGN(self.config, dataset)
        weight_file = self.getFileName(graph)
        model.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        model = model.eval()

        with torch.no_grad():
            GPU = torch.cuda.is_available()
            device = torch.device('cuda' if GPU else "cpu")
            batch_users_gpu = torch.tensor([user_id], dtype=torch.long).to(device)
            rating = model.getUsersRating(batch_users_gpu)
            allPos = dataset.getUserPosItems([user_id])
            exclude_items = [item for items in allPos for item in items]
            rating[:, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=k)
        return rating_K


    def getFileName(self, graph):
        
        ROOT_PATH = "./"
        FILE_PATH = join(ROOT_PATH, 'checkpoints')
        file = f"{'SocialLGN'}-{graph}-{self.config['layer']}layer-" \
                f"{self.config['latent_dim_rec']}.pth.tar"
    #/root/Graph_Toolformer/Graph_Toolformer_Package/koala/graph_models/socialLGN/local_data/PreTrained_GraphBert/lastfm/SocialLGN-lastfm-3layer-64.pth.tar   
            
        return os.path.join(FILE_PATH, file)

if __name__ == "__main__":
    model = SocialLGNModel()
    result = model.topk_recommendation('lastfm', (701, 10))
    print(result)