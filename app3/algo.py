import faiss
import numpy as np
import logging
import pickle5 as pickle
import os

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger('itemrep-inference')

artifact_path = 'app3/assets/data/artifacts'

# logging.info("Loading item vectors")
item_vectors = pickle.load(open(os.path.join(artifact_path,"item_vectors.p"), "rb"))

# logging.info("Loading indexer")
index = faiss.read_index(os.path.join(artifact_path,"vector.index"))

def topk_similar(ivec, topk=5):
    _, I = index.search(ivec.reshape(1,-1), topk)
    return [list(item_vectors.keys())[i] for i in I.flatten()]