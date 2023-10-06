# -*- coding: utf-8 -*-

from .embed import HFEmbeddings
from .db import Chroma

class Text2vec:
    def __init__(self, 
                 cache_folder='models',
                 model_kwargs={'device': 'cpu'}):
                  
        self.embed = HFEmbeddings(cache_folder=cache_folder, 
                         model_kwargs=model_kwargs)
                         
    def get_db(self, name):
            
        db = Chroma(persist_directory=name, 
                    embedding_function=self.embed)
        return db   
        
