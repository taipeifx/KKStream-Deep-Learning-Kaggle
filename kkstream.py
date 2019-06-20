import csv
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense
import sklearn.metrics

#
def build_model():
    model = Sequential()
    model.add(Dense(32, activation='elu', input_shape=(896,)))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(28, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    return model

#
def write_result(name, predictions):
    if predictions is None:
        raise Exception('need predictions')

    predictions = predictions.flatten()

    if not os.path.exists('./results/'):
        os.makedirs('./results/')

    path = os.path.join('./results/', name)

    with open(path, 'wt', encoding='utf-8', newline='') as csv_target_file:
        target_writer = csv.writer(csv_target_file, lineterminator='\n')

        header = [
            'user_id',
            'time_slot_0', 'time_slot_1', 'time_slot_2', 'time_slot_3',
            'time_slot_4', 'time_slot_5', 'time_slot_6', 'time_slot_7',
            'time_slot_8', 'time_slot_9', 'time_slot_10', 'time_slot_11',
            'time_slot_12', 'time_slot_13', 'time_slot_14', 'time_slot_15',
            'time_slot_16', 'time_slot_17', 'time_slot_18', 'time_slot_19',
            'time_slot_20', 'time_slot_21', 'time_slot_22', 'time_slot_23',
            'time_slot_24', 'time_slot_25', 'time_slot_26', 'time_slot_27',
        ]

        target_writer.writerow(header)

        for i in range(0, len(predictions), 28):
            # NOTE: 57159 is the offset of user ids
            userid = [57159 + i // 28]
            labels = predictions[i:i+28].tolist()

            target_writer.writerow(userid + labels)

#
dataset = np.load('./datasets/v0_eigens.npz')

train_data_size = dataset['train_eigens'].shape[0]
valid_data_size = train_data_size // 5
train_data_size = train_data_size - valid_data_size
indices = np.arange(train_data_size + valid_data_size)

train_data = dataset['train_eigens'][indices[:train_data_size]]
valid_data = dataset['train_eigens'][indices[train_data_size:]]

train_eigens = train_data[:, :-28]
train_labels = train_data[:, -28:]
valid_eigens = valid_data[:, :-28]
valid_labels = valid_data[:, -28:]
issue_eigens = dataset['issue_eigens'][:, :-28]

print('train_eigens.shape = {}'.format(train_eigens.shape))
print('train_labels.shape = {}'.format(train_labels.shape))
print('valid_eigens.shape = {}'.format(valid_eigens.shape))
print('valid_labels.shape = {}'.format(valid_labels.shape))

#
model = build_model()

model.fit(
    x=train_eigens,
    y=train_labels,
    batch_size=450,
    epochs=11,
    verbose=2,
    validation_data=(valid_eigens, valid_labels),
    shuffle=True)

#
def auc(guess, truth):
    """
    """
    guess = guess.flatten()
    truth = truth.flatten()
    
    fprs, tprs, _ = sklearn.metrics.roc_curve(truth, guess)

    return sklearn.metrics.auc(fprs, tprs)

valid_guesss = model.predict(valid_eigens)
valid_guesss_auc = auc(valid_guesss, valid_labels)
print ('valid_guesss_auc = {}'.format(valid_guesss_auc))

#
issue_guesss = model.predict(issue_eigens)
write_result('predict_dense.csv', issue_guesss)