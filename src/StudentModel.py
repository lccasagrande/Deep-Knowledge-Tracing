from keras.callbacks import ModelCheckpoint, Callback, CSVLogger, Progbar
from keras.models import Sequential
from keras.layers import TimeDistributed, Masking, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras import backend as K
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import random
import math
import numpy as np


# This method is for internal use. You should not use it outside of this file.
def model_evaluate(test_gen, model, metrics, verbose=0):
    def predict():
        def get_target_skills(preds, labels):
            target_skills = labels[:, :, 0:test_gen.num_skills]
            target_labels = labels[:, :, test_gen.num_skills]

            target_preds = np.sum(preds * target_skills, axis=2)

            return target_preds, target_labels

        y_true_t = []
        y_pred_t = []
        test_gen.reset()

        while not test_gen.done:
            # Get batch
            batch_features, batch_labels = test_gen.next_batch()

            # Predict
            predictions = model.predict_on_batch(batch_features)

            # Get target skills
            target_preds, target_labels = get_target_skills(predictions, batch_labels)
            flat_pred = np.reshape(target_preds, [-1])
            flat_true = np.reshape(target_labels, [-1])

            # Remove mask
            mask_idx = np.where(flat_true == -1.0)[0]
            flat_pred = np.delete(flat_pred, mask_idx)
            flat_true = np.delete(flat_true, mask_idx)

            # Save it
            y_true_t.extend(flat_true)
            y_pred_t.extend(flat_pred)

            if verbose and test_gen.step < test_gen.total_steps:
                progbar.update(test_gen.step)

        return y_true_t, y_pred_t

    assert (isinstance(test_gen, DataGenerator))
    assert (model is not None)
    assert (metrics is not None)

    if verbose:
        print("==== Evaluation Started ====")

    progbar = Progbar(target=test_gen.total_steps, verbose=verbose)

    y_true, y_pred = predict()

    bin_pred = [1 if p > 0.5 else 0 for p in y_pred]

    results = {}
    if 'auc' in metrics:
        results['auc'] = roc_auc_score(y_true, y_pred)
    if 'acc' in metrics:
        results['acc'] = accuracy_score(y_true, bin_pred)
    if 'pre' in metrics:
        results['pre'] = precision_score(y_true, bin_pred)

    if verbose:
        progbar.update(test_gen.step, results.items())
        print("==== Evaluation Done ====")

    return results

# This class is for internal use. You should not use it outside of this file.
class MetricsCallback(Callback):
    def __init__(self, data_gen, metrics, verbose=0):
        super(MetricsCallback, self).__init__()
        assert (isinstance(data_gen, DataGenerator))
        assert (metrics is not None)

        self.data_gen = data_gen
        self.metrics = metrics
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        if 'auc' in self.metrics:
            self.params['metrics'].append('val_auc')
        if 'acc' in self.metrics:
            self.params['metrics'].append('val_acc')
        if 'pre' in self.metrics:
            self.params['metrics'].append('val_pre')

    def on_epoch_end(self, epoch, logs={}):
        results = model_evaluate(self.data_gen, self.model, metrics=['auc', 'acc', 'pre'], verbose=self.verbose)

        if 'auc' in self.metrics:
            logs['val_auc'] = results['auc']
        if 'acc' in self.metrics:
            logs['val_acc'] = results['acc']
        if 'pre' in self.metrics:
            logs['val_pre'] = results['pre']

# This class defines the DKT model.
class DKTModel(object):
    def __init__(self, num_skills, num_features, optimizer='rmsprop', hidden_units=100, batch_size=5, dropout_rate=0.5):
        def get_target_skills(y_true, y_pred):
            target_skills = y_true[:, :, 0:num_skills]
            target_labels = y_true[:, :, num_skills]
            target_preds = K.sum(y_pred * target_skills, axis=2)

            return target_preds, target_labels

        def loss_function(y_true, y_pred):
            target_preds, target_labels = get_target_skills(y_true, y_pred)
            return K.binary_crossentropy(target_labels, target_preds)

        self.batch_size = batch_size
        self.num_skills = num_skills

        self.__model = Sequential()
        self.__model.add(Masking(-1., batch_input_shape=(batch_size, None, num_features)))
        self.__model.add(LSTM(hidden_units, return_sequences=True, stateful=True))
        self.__model.add(Dropout(dropout_rate))
        self.__model.add(TimeDistributed(Dense(num_skills, activation='sigmoid')))
        self.__model.compile(loss=loss_function, optimizer=optimizer)

    def load_weights(self, filepath):
        assert(filepath is not None)
        self.__model.load_weights(filepath)

    def fit(self, train_gen, epochs, val_gen, verbose=0, filepath_bestmodel=None, filepath_log=None):
        assert (isinstance(train_gen, DataGenerator))
        assert (isinstance(val_gen, DataGenerator))

        callbacks = []
        callbacks.append(MetricsCallback(val_gen, metrics=['auc','pre','acc']))

        if filepath_bestmodel is not None:
            callbacks.append(ModelCheckpoint(filepath_bestmodel, monitor='val_loss', verbose=verbose, save_best_only=True))
        if filepath_log is not None:
            callbacks.append(CSVLogger(filepath_log))

        if verbose:
            print("==== Training Started ====")

        history = self.__model.fit_generator(shuffle=False,
                                             validation_data=val_gen.get_generator(),
                                             validation_steps=val_gen.total_steps,
                                             epochs=epochs,
                                             steps_per_epoch=train_gen.total_steps,
                                             generator=train_gen.get_generator(),
                                             callbacks=callbacks,
                                             verbose=verbose)

        if verbose:
            print("==== Training Done ====")

        return history

    def evaluate(self, test_gen, metrics, verbose=0, filepath_log=None):
        assert (isinstance(test_gen, DataGenerator))
        assert (metrics is not None)

        results = model_evaluate(test_gen, self.__model, metrics, verbose)

        if filepath_log is not None:
            with open(filepath_log, 'w') as fl:
                fl.write("auc,acc,pre\n{0},{1},{2}".format(results['auc'], results['acc'], results['pre']))

        return results


# This class is responsible for feeding the data into the model following a specific format.
class DataGenerator(object):
    def __init__(self, features, labels, num_skills, batch_size):
        self.features = features
        self.labels = labels
        self.num_skills = num_skills
        self.batch_size = batch_size

        self.step = 0
        self.done = False
        self.feature_dim = num_skills * 2
        self.label_dim = num_skills + 1
        self.features_len = len(features)
        self.total_steps = int(math.ceil(float(self.features_len) / self.batch_size))
        self.feature_encoder = OneHotEncoder(self.feature_dim, sparse=False)
        self.label_encoder = OneHotEncoder(self.label_dim, sparse=False)

    # Ref: https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
    def __pad_sequences(self, sequences, maxlen=None, dim=1, dtype='int32', padding='pre', truncating='pre', value=0.):
        '''
            Override keras method to allow multiple feature dimensions.

            @dim: input feature dimension (number of features per timestep)
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x

    def next_batch(self):
        def fill_batches(x, y):
            for e in range(self.batch_size - len(x)):
                x.append([np.array([-1.0 for _ in range(0, self.feature_dim)])])
                y.append([np.array([-1.0 for _ in range(0, self.label_dim)])])

            return x, y

        def pad_sequences(x, y):
            max_seq_steps = max([len(seq) for seq in x])
            x = self.__pad_sequences(x, padding='pre', maxlen=max_seq_steps, dim=self.feature_dim, value=-1.0, dtype='float')
            y = self.__pad_sequences(y, padding='pre', maxlen=max_seq_steps, dim=self.label_dim, value=-1.0, dtype='float')

            return x, y

        def encode_batch(batch_questions, batch_answers):
            x = []
            y = []
            for idx, questions in enumerate(batch_questions):
                x_student = []
                y_student = []

                x_data = np.zeros(self.feature_dim, dtype=int)
                answers = batch_answers[idx]

                for skill_index, skill_value in enumerate(questions):
                    answer = answers[skill_index]

                    # Encode skill_id
                    x_student.append(x_data)
                    skill_answer = skill_value * 2 + answer
                    x_data = self.feature_encoder.fit_transform(np.reshape(skill_answer, (-1, 1)))
                    x_data = np.squeeze(x_data)

                    # Encode label
                    y_data = self.label_encoder.fit_transform(np.reshape(skill_value, (-1, 1)))
                    y_data = np.squeeze(y_data)
                    y_data[-1] = answer
                    y_student.append(y_data)

                x.append(x_student)
                y.append(y_student)

            return x, y

        assert(~self.done)

        start_pos = self.step * self.batch_size
        end_pos = (self.step + 1) * self.batch_size

        if end_pos >= self.features_len:
            self.done = True
            end_pos = self.features_len

        # Apply one-hot encoding
        x_batch, y_batch = encode_batch(self.features[start_pos:end_pos], self.labels[start_pos:end_pos])

        # Fill up incomplete batch
        x_batch, y_batch = fill_batches(x_batch, y_batch)

        # Pad sequences to the same size
        x_batch, y_batch = pad_sequences(x_batch, y_batch)

        self.step += 1

        return x_batch, y_batch

    def reset(self, shuffle=True):
        if shuffle:
            self.shuffle()

        self.done = False
        self.step = 0

    def shuffle(self):
        combined = list(zip(self.features, self.labels))
        random.shuffle(combined)
        self.features[:], self.labels[:] = zip(*combined)

    def get_generator(self):
        while True:
            self.reset()
            while not self.done:
                batch_features, batch_labels = self.next_batch()
                yield batch_features, batch_labels
