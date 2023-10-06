import time   
import text2vec as tv
import argparse

pq = tv.Text2Vec()

def create_db():
    start = time.time()    
    db = pq.get_db(name='db_test')
    print( time.time() - start)
    start = time.time()
    texts = ['ok1', 'ok2', 'ok3']
    metas = [{'cn':1}, {'cn':2}, {'cn':3}]
    ids = db.add_texts(texts, metas)
    print(ids)
    print( time.time() - start)

def query_db():
    start = time.time()    
    db = pq.get_db(name='db_test')
    print( time.time() - start)
    val = db.similarity_search_with_score('ok1')
    print(val)
    print( time.time() - start)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 0)')    
    args = parser.parse_args()
    if args.m == 1:
        create_db()
    elif args.m == 2:
        query_db()
    