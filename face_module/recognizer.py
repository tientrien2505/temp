from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image as img_keras
import pickle
import cv2
import nmslib
import numpy as np
from .searcher import search_nmslib_index
import os


class DistancesVoting():
    def __init__(self, index_path, data_embedded, dist_threshold, k_nearests):
        self.index = nmslib.init(method='hnsw', space='cosinesimil', data_type=nmslib.DataType.DENSE_VECTOR)
        self.index.loadIndex(index_path, load_data=True)
        self.dist_threshold = dist_threshold
        self.data_labels = data_embedded['ids']
        self.unknown_id = 0
        self.k_nearests = k_nearests

    def get_final_predictions(self, predicted_labels, dists):
        a =  np.zeros(np.max(self.data_labels) + 1) 

        for i,label in enumerate(predicted_labels):
                a[label]+=1.0/(i+dists[i])
        scores=sorted(a[a>0.0])

        ranked_labels=sorted(range(len(a)), key=a.__getitem__)
        print('ranked_labels', predicted_labels)

        candidates =ranked_labels[-len(scores):]
        candidates =candidates[::-1]

        temp_index = np.where(np.array(predicted_labels) == candidates[0])[0]
        predicted_dists = np.array(dists)[temp_index]
        prop = 1- np.mean(predicted_dists)/0.54
        final_id = candidates[0]
        print('candidates', candidates)
        if prop < 0.5:
            final_id = self.unknown_id

        return final_id, prop

    def vote_distances_2(self, distances, f_ids, Y_train):
        D = np.array(distances)
        I = np.array(f_ids) 
        predictions = []
        dist_list = []
        for k in range(len(I)):
                la = int(Y_train[I[k]])
                if 1==1:
                    dis = D[k]
                    dist_list.append(dis)
                    if dis > self.dist_threshold :
                        predictions.append(self.unknown_id)
                    else:
                        predictions.append(la)
        prediction, prop = self.get_final_predictions(predictions, dist_list)
        return prediction, prop

    def predict(self, embeddings):

        query_results = search_nmslib_index(self.index, embeddings, self.k_nearests)
        print('query_results', query_results)
        labels = []
        props = []
        for i, result in enumerate(query_results):
            distances = result[1]
            f_ids = result[0]
            name, prop = self.vote_distances_2(distances, f_ids, self.data_labels)
            labels.append(name)
            props.append(prop)
        print('labels', labels)
        print('props', props)
        return labels, props

    def predict_simple(self, embeddings):
        labels, distances = self.index.knnQuery(embeddings, k=10)
        print('labels', labels)
        print('distances', distances)

        props = [1- distances[0]]

        if props[0] < 0.5:
            labels = [self.unknown_id]

        return labels, props


class FaceRecognizer():
    def __init__(self, index_path= './model/nms_index', embedding_path= './model/data_embedded.pickle',  dist_threshold=0.24):
        if os.path.isfile(embedding_path):
            data = pickle.loads(open(embedding_path, "rb").read())
            if os.path.isfile(index_path):
                self.predictor = DistancesVoting(index_path, data, dist_threshold, 7)
                self.unknown_id = self.predictor.unknown_id
            else:
                print ("MISSING INDEX_PATH !")
        else:
            print ("MISSING EMBEDDING_PATH !")
        
        self.vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


    def get_embedding_vggface(self, face_img):
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        face_img = img_keras.img_to_array(face_img)
        samples = np.expand_dims(face_img, axis=0)
        samples = preprocess_input(samples, version=2)
        print('samples', samples.shape)
        emb = self.vgg_model.predict(samples)
        return emb


    def recognize(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        embedding = self.get_embedding_vggface(face_img)[0]
        print('embedding', embedding.shape)
        ids, props = self.predictor.predict(np.array([embedding]))
        id = ids[0] if ids[0] != self.unknown_id else 0

        return id, props[0]

