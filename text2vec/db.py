# -*- coding: utf-8 -*-
import os
import chromadb
import uuid
from chromadb.config import Settings
import copy
import math
import numpy as np

class Chroma:

    _DEFAULT_COLLECTION_NAME = "mygpt"  
        
    def __init__(self, embedding_function, persist_directory):
        
        self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False))                
        self._embedding_function = embedding_function
        self._collection = self._client.get_or_create_collection(
            name=self._DEFAULT_COLLECTION_NAME,
            embedding_function=self._embedding_function.embed_documents,
            metadata=None,
        )
    
    def add_texts(self, texts, metadatas=None, ids=None, **kwargs):
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
            print(len(embeddings), len(embeddings[0]))
        self._collection.add(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids

    def delete_collection(self):
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def get(self):
        return self._collection.get()

    def persist(self):       
        self._client.persist()        
        
    def query_collection(self, query_texts=None, query_embeddings=None, n_results=10, 
                         where=None, **kwargs):
        
        for i in range(n_results, 0, -1):
            try:
                return self._collection.query(
                    query_texts=query_texts,
                    query_embeddings=query_embeddings,
                    n_results=i,
                    where=where,
                    **kwargs,
                )
            except chromadb.errors.NotEnoughElementsException:
                pass
        raise chromadb.errors.NotEnoughElementsException(
            f"No documents found for Chroma collection {self._collection.name}"
        )
    
    def cos_sim(self, vec1, vec2):
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
        cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cos_sim
        
    # y = ( x  +(n-1)*mean ) / n
    # x = n *(y - mean ) + mean    
    def modify_dis(self, dis, query, word, ma=0.6):
        num = len(word) // len(query)
        if num <= 1: num = 1
        ma = dis * ma
        return math.sqrt(num) *(dis - ma) + ma
        
    def similarity_search_with_score(self, query, k=30,
                                     filter=None, **kwargs):
        query_embedding = self._embedding_function.embed_query(query)
        results = self.query_collection(
                query_embeddings=[query_embedding], n_results=k, where=filter,
                include=["embeddings", "metadatas", "documents", "distances"]
            )
            
        ans = []        
        for dis, one, word, meta in zip(results["distances"][0], results["embeddings"][0],
                 results["documents"][0], results["metadatas"][0]):            
            cos = self.cos_sim(query_embedding, one)
            v = self.modify_dis(cos, query, word)
            ans.append([dis, cos, v, word, meta])            
        ans = np.array(ans)
        idx = np.argsort(ans[:, 1])
        return ans[idx[::-1]]
        
if __name__ == "__main__":
    pass
    
    
    