
import random
import math

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder


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
                fl.write("auc,acc,pre\n{0},{1},{2}".format(
                    results['auc'], results['acc'], results['pre']))

        return results


def load_dataset(fn, batch_size=32):
    df = pd.read_csv(fn, dtype={'skill_name': str})

    # Step 1 - Remove questions without skill
    df.dropna(subset=['skill_id'], inplace=True)

    # Step 2 - Enumerate skill id
    df['skill'], _ = pd.factorize(df['skill_id'], sort=True)

    # Step 3 - Cross skill id with answer to form a synthetic feature
    df['skill_with_answer'] = df['skill'] * 2 + df['correct']

    # Step 4 - Convert to a sequence per user id
    seq = df.groupby('user_id').apply(
        lambda r: (
            r['skill_with_answer'].values,
            r['skill'].values,
            r['correct'].values
        )
    )

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32),
        output_shapes=((None,), (None,), (None,))
    )

    # Step 6 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(-1, -1, 0.),
        padded_shapes=([None], [None], [None])
    )

    # Step 7 - Encode categorical features
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
        lambda f, m, l: (
            tf.one_hot(f, depth=features_depth),
            tf.one_hot(m, depth=skill_depth),
            l)
    )

    return dataset
