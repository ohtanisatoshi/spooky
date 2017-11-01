import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

filename = 'spooky.pkl'
with open(filename, 'rb') as f:
    vector_rank_list = pickle.load(f)

vector_list = [r['tv'] for r in vector_rank_list]
rank_list = [r['r'] for r in vector_rank_list]

vector_array = np.array(vector_list)
vector_array_norm = np.array(vector_array.shape)
vector_array_norm = np.divide(vector_array, np.sqrt(np.sum(vector_array**2, axis=1)).reshape(-1, 1))

pca = PCA(n_components=2)
pca.fit(vector_array_norm)
vector_array_transformed = pca.fit_transform(vector_array_norm)
#model = TSNE(n_components=2, perplexity=50, n_iter=500, verbose=3, random_state=1)
#vector_array_transformed = model.fit_transform(vector_array_norm)

a_x = []
a_y = []
b_x = []
b_y = []
c_x = []
c_y = []
for i, (x, y) in enumerate(vector_array_transformed):
    if rank_list[i] == 'EAP':
        a_x.append(x)
        a_y.append(y)
    elif rank_list[i] == 'HPL':
        b_x.append(x)
        b_y.append(y)
    elif rank_list[i] == 'MWS':
        c_x.append(x)
        c_y.append(y)
    else:
        pass

plt.scatter(a_x, a_y, c='r', marker='o', label='EAP')
plt.scatter(b_x, b_y, c='y', marker='^', label='HPL')
plt.scatter(c_x, c_y, c='b', marker='v', label='MWS')
plt.legend(loc='upper right')
plt.show()
print('done')