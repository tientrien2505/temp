import nmslib
import numpy as np

def create_nmslib_index(embeddings, output_path=None, save_index=True):
    index_time_params = {'M': 15, 'indexThreadQty': 4, 'efConstruction': 100, 'post' : 2}
    embeddings = np.array(embeddings)
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
    query_results = index.knnQueryBatch(embeddings, k = k_nearests, num_threads=5)
    return query_results

