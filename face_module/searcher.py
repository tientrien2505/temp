import faiss
import os
import cv2
import nmslib
import numpy as np

class ExactIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension,)
        self.index.add(self.vectors)
        
    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k) 
        # I expect only query on one vector thus the slice
        return [self.labels[i] for i in indices[0]]

def create_faiss_index(embeddings, emb_dim, output_path=None, save_index=True):
    # index = faiss.IndexFlatL2(emb_dim)
    index = faiss.IndexHNSWFlat(emb_dim, 64)
    index.add(embeddings)
    # embeddings = np.array(embeddings)
    # faiss.normalize_L2(embeddings)
    # ids = np.array(list(map(lambda x: ids.index(x) , ids)))
    # print(index.is_trained)


    # index = faiss.IndexIDMap(index_init)
    # index.add_with_ids(embeddings, ids)


    # ncentroids = 4
    # quantizer = faiss.IndexFlatL2(128)
    # index = faiss.IndexIVFFlat(quantizer, 128, ncentroids, faiss.METRIC_L2)
    # index.train(embeddings)

    # index = faiss.IndexIDMap(index)
    # index.add_with_ids(embeddings, ids)

    if save_index:
        faiss.write_index(index, output_path)
    else:
        return index


#Create Index
# flower_index = faiss.read_index("face_faiss_128.bin")
def search_faiss_index(index, embedding, k_nearests):
    # faiss.normalize_L2(embedding)
    f_dists, f_ids = index.search(embedding, k=k_nearests)
    results = list(zip(f_ids, f_dists))
    return results


def create_nmslib_index(embeddings, output_path=None, save_index=True):
    index_time_params = {'M': 15, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 2}
    embeddings = np.array(embeddings)
    # faiss.normalize_L2(embeddings)
    index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(embeddings)
    index.createIndex(index_time_params, print_progress=True)
    if save_index:
        index.saveIndex(output_path, save_data=True)
    else:
        return index


def search_nmslib_index(index, embeddings, k_nearests):
    query_time_params = {'efSearch': 100}
    index.setQueryTimeParams(query_time_params)
    # print(embeddings)
    # faiss.normalize_L2(embeddings)
    query_results = index.knnQueryBatch(embeddings, k = k_nearests, num_threads=5)
    return query_results


# if __name__ == '__main__':
    # from face_encoder import get_embedding_FaceNet
    # from keras.models import load_model
    # from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
    # import pickle
    # import time
    # import sys
    # sys.path.append("..")
    # from detector import *
    # from sklearn.model_selection import train_test_split

    
    # create_new = True

    # test_img_path = '/media/sonlh/F2123E34123DFE63/face_data_160/CuongBC/CuongBC_4.jpg'
    # embeddings_path = 'embeddings/data_embedded_vggface.pickle'
    # output_path = 'search_index/face_2048_nmslib_index'
    # start_time = time.time()
    # f = open(embeddings_path, "rb")
    # data = pickle.loads(f.read())
    # X = np.array(data['encodings'], dtype='float32')
    # Y = np.array(data['names'])

    # # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    # print("--- load embedding: %s seconds ---" % (time.time() - start_time))
    # k = len(np.unique(data['names']))


    # if create_new == True:
    #     # start_time = time.time()
    #     # create_faiss_index(data['encodings'], data['names'], output_path)
    #     # print("--- %s seconds ---" % (time.time() - start_time))
    #     print(data)


    #     start_time = time.time()
    #     create_nmslib_index(data['encodings'], output_path)
    #     print("--- %s seconds ---" % (time.time() - start_time))
    
    # else:



    #     img = cv2.imread(test_img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     FN_model = load_model('facenet_keras.h5')
    #     embedding = get_embedding_FaceNet(FN_model, img)
    #     print(embedding.shape)

    #     faiss.normalize_L2(embedding)

    #     index = faiss.read_index("face_faiss_128_IVF.bin")
    #     print('index: ', index)


    #     # Append new face
    #     # face_detector = haar_cascade_detector()
    #     # person_path = '/home/sonlh/Pictures/H_Long'
    #     # image_paths = os.listdir(person_path)
    #     # for path in image_paths:
    #     #     image = cv2.imread(f'{person_path}/{test_img_path}')
    #     #     face_loc = face_detector.detect(image)


    #     # start_time = time.time()
    #     # index.add_with_ids(embedding, 4)
    #     # faiss.write_index(index, "face_faiss_128.bin")
        
    #     # data['names'].append('1_QuyVD_special')
    #     # np.append(data['encodings'], embedding[0])

    #     # with open(embeddings_path,'wb') as wfp:
    #     #     pickle.dump(data, wfp)
    #     # print("--- add index: %s seconds ---" % (time.time() - start_time))
        



    #     start_time = time.time()

    #     # result = index.search(embedding, 4)

    #     index = nmslib.init(method='hnsw', space='cosinesimil')
    #     index.loadIndex('face_128_nmslib_index', load_data=True)
    #     ids, distances = index.knnQuery(embedding, k=10)        
        
    #     print("--- %s secondcs ---" % (time.time() - start_time))
        
        
    #     print(np.array(data['names'])[ids])
    #     print(distances)


    #     # ids = list(map(lambda x: data['names'][int(x)], f_ids[0].tolist()))
    #     # print(ids)