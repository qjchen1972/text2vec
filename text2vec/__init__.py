# -*- coding: utf-8 -*-

from .embed import HFEmbeddings
from .db import Chroma

def init_db(name, 
            cache_folder='models',
            model_kwargs={'device': 'cpu'}):
            
    embed = HFEmbeddings(cache_folder=cache_folder, 
                         model_kwargs=model_kwargs)
    db = Chroma(persist_directory=name, embedding_function=embed)
    return db    
