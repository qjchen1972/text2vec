#https://huggingface.co/sentence-transformers/all-mpnet-base-v2
import sentence_transformers

class EncodeModel:

    def __init__(self, cache_folder=None, model_kwargs=dict()):
        model_name = 'all-MiniLM-L6-v2'
        self.client = sentence_transformers.SentenceTransformer(
                    model_name,
                    cache_folder=cache_folder,
                    **model_kwargs,
                    )
    
    def encode(self, texts, **encode_kwargs):
        return self.client.encode(texts, **encode_kwargs)
    
        