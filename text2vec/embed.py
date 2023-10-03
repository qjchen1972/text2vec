# -*- coding: utf-8 -*-
import os
#from encode.mpnet import EncodeModel 
from .encode.text2vec import EncodeModel 



class HFEmbeddings:   
    
    def __init__(self, 
                 cache_folder='models', 
                 model_kwargs={'device': 'cpu'}, 
                 encode_kwargs=dict()):
                 
        cache_folder = os.path.join(os.path.dirname(__file__), 
                                    cache_folder)        
        self.client = EncodeModel(cache_folder=cache_folder, 
                                  model_kwargs=model_kwargs) 
        self.encode_kwargs = encode_kwargs        
    
    def embed_documents(self, texts):
        texts = list(map(lambda x: x.replace("\n", " "), texts))        
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text):
        text = text.replace("\n", " ")
        embedding = self.client.encode(text, **self.encode_kwargs)
        return embedding.tolist()
        
if __name__ == "__main__":

    import time    
    cache_folder = 'models'
    model_kwargs = {'device': 'cpu'}    
    start = time.time()
    embeddings = HFEmbeddings(cache_folder=cache_folder, 
                              model_kwargs=model_kwargs)
    print(time.time() - start)
    
    val = embeddings.embed_query('ok')
    
    print(time.time() - start, len(val))
    
    