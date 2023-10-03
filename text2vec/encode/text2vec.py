#https://huggingface.co/GanymedeNil/text2vec-large-chinese
import sentence_transformers

class EncodeModel:

    def __init__(self, cache_folder=None, model_kwargs=dict()):
        #model_name = 'shibing624/text2vec-large-chinese'
        model_name = 'GanymedeNil/text2vec-large-chinese'
        self.client = sentence_transformers.SentenceTransformer(
                    model_name,
                    cache_folder=cache_folder,
                    **model_kwargs,
                    )
    
    def encode(self, texts, **encode_kwargs):
        return self.client.encode(texts, **encode_kwargs)
    
        