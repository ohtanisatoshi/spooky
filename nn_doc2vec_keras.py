import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
import pickle
from sklearn.cross_validation import train_test_split
import time
import os

def get_rank_one_hot(rank):
    rank_vector = []
    if rank == 'EAP':
        rank_vector = [1.0, 0.0, 0.0]
    elif rank == 'HPL':
        rank_vector = [0.0, 1.0, 0.0]
    elif rank == 'MWS':
        rank_vector = [0.0, 0.0, 1.0]
    else:
        print('data error!')
        exit(1)

    return rank_vector


def load_data(picklefile_name):
    with open(picklefile_name, 'rb') as f:
        test_data = pickle.load(f)

    return test_data


def get_normed_vector(data_rec):
    # doc2vec vector
    dv_array = np.array(data_rec['dv'])
    #dv_array_norm = np.divide(dv_array, np.sqrt(np.sum(dv_array**2)))
    dv_array_norm = dv_array - np.min(dv_array)
    dv_array_norm = dv_array_norm / np.max(dv_array_norm)
    # tfidf vectof
    tv_array = np.array(data_rec['tv'])
    tv_array_norm = tv_array - np.min(tv_array)
    tv_array_norm = tv_array_norm / np.max(tv_array_norm)

    #return np.concatenate((dv_array_norm, tv_array_norm))
    return tv_array_norm


def load_train_data(picklefile_name):
    test_data = load_data((picklefile_name))

    X_train_work = []
    y_train_work = []
    filename_list = []
    for r in test_data:
        if len(r['r']) == 0:
            continue

        normed_vector = get_normed_vector(r)

        X_train_work.append(normed_vector)
        y_train_work.append(get_rank_one_hot(r['r']))
        filename_list.append(r['f'])

    X_train = np.array(X_train_work)
    y_train = np.array(y_train_work)

    return X_train, y_train, filename_list


def load_test_data(picklefile_name):
    test_data = load_data((picklefile_name))

    X_train_work = []
    filename_list = []
    for r in test_data:
        if len(r['r']) > 0:
            continue

        normed_vector = get_normed_vector(r)

        X_train_work.append(normed_vector)
        filename_list.append(r['f'])

    X_train = np.array(X_train_work)

    return X_train, filename_list


def define_model(input_size):
    model = Sequential()
    model.add(Dense(1000, input_dim=input_size))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    return model

def train_nn(X_all, y_all, filename_list, modelfile_name):
    x_index = np.random.permutation(X_all.shape[0])
    y_index = np.random.permutation(X_all.shape[0])  # dummy
    x_train_index, x_test_index, _, _ = train_test_split(x_index, y_index, train_size=0.9, random_state=None)
    X_train = X_all[x_train_index]
    X_test = X_all[x_test_index]
    y_train = y_all[x_train_index]
    y_test = y_all[x_test_index]
    filename_list_for_train = np.array(filename_list)[x_train_index]
    filename_list_for_test = np.array(filename_list)[x_test_index]

    input_size = X_all.shape[1]

    model = define_model(input_size)

    st = time.time()
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    #model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
    model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
    print('took {:.2f}'.format(time.time()-st))

    # save model
    model.save(modelfile_name)

    st = time.time()
    result_prob = model.predict_proba(X_test, batch_size=5)

    ranks = ['EAP', 'HPL', 'MWS']
    a_cnt = 0
    b_cnt = 0
    c_cnt = 0
    correct_cnt = 0
    with open('result.csv', 'w') as f:
        for result, t, filename in zip(result_prob, y_test, filename_list_for_test):
            #result = sess.run(y_conv_softmax, feed_dict=feed_dict)
            answer_max_index = np.argmax(t)
            answer_rank = ranks[answer_max_index]
            result_max_index = np.argmax(result)
            result_rank = ranks[result_max_index]
            output = '{},{},{:.4f},{:.4f},{:.4f},{}'.format(
                answer_rank, result_rank, result[0], result[1], result[2], filename)
            f.write(output)
            f.write(output)
            file_text = ''
            if answer_max_index == result_max_index:
                correct_cnt += 1
            else:
                with open(os.path.join('/home/satoshi/dev/kaggle/spooky/data/text', filename + '.txt')) as ff:
                    file_text = '"{}"'.format(ff.read())
            f.write(',{}\n'.format(file_text))

            if answer_rank == 'EAP':
                a_cnt += 1
            elif answer_rank == 'HPL':
                b_cnt += 1
            elif answer_rank == 'MWS':
                c_cnt += 1

            print(output)

    print('EAP: {} HPL: {} : MWS{}'.format(a_cnt, b_cnt, c_cnt))
    print('Accuracy: {:.2f} %'.format(correct_cnt / len(y_test) * 100))

    print('took {:.2f}'.format(time.time()-st))


def predict(X_all, filename_list, modelfile_name):
    model = load_model(modelfile_name)

    st = time.time()
    result_prob = model.predict_proba(X_all, batch_size=5)

    with open('test_result.csv', 'w') as f:
        f.write('id,EAP,HPL,MWS\n')
        for result, filename in zip(result_prob, filename_list):
            id = filename.split('_')[0]
            output = '{},{:.2f},{:.2f},{:.2f}\n'.format(id, result[0], result[1], result[2])
            f.write(output)

    print('took {:.2f}'.format(time.time()-st))

def main(picklefile_name='spooky.pkl', modelfile_nmae='model.h5', train=True):
    if train:
        X_all, y_all, filename_list = load_train_data(picklefile_name)
        train_nn(X_all, y_all, filename_list, modelfile_nmae)
    else:
        X_all, filename_list = load_test_data(picklefile_name)
        predict(X_all, filename_list, modelfile_nmae)


if __name__ == '__main__':
    train_mode = True
    main(train=train_mode)
