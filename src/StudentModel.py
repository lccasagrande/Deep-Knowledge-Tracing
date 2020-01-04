import tensorflow as tf
import pandas as pd
import numpy as np


def target_skill_binary_crossentropy(y_true, y_pred):
    # Get skills and labels from y_true
    skills, labels = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    skill_preds = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return tf.keras.losses.binary_crossentropy(labels, skill_preds)


class DKTModel(tf.keras.Model):
    def __init__(self, num_features, num_skills, hidden_units=100, dropout_rate=0.2):
        inputs = tf.keras.Input(shape=(None, num_features), name='inputs')

        x = tf.keras.layers.Masking(mask_value=0.)(inputs)

        x = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(x)

        dropout = tf.keras.layers.Dropout(dropout_rate)
        x = tf.keras.layers.TimeDistributed(dropout)(x)

        dense = tf.keras.layers.Dense(num_skills, activation='sigmoid')
        outputs = tf.keras.layers.TimeDistributed(dense, name='outputs')(x)

        super(DKTModel, self).__init__(inputs=inputs,
                                       outputs=outputs,
                                       name="DKTModel")

    def compile(self, optimizer, metrics=None):
        """Configures the model for training.
        Arguments:
            optimizer: String (name of optimizer) or optimizer instance.
                See `tf.keras.optimizers`.
            metrics: List of metrics to be evaluated by the model during training
                and testing. Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary, such as
                `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                You can also pass a list (len = len(outputs)) of lists of metrics
                such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                `metrics=['accuracy', ['accuracy', 'mse']]`.
        Raises:
            ValueError: In case of invalid arguments for
                `optimizer` or `metrics`.
        """
        super(DKTModel, self).compile(
            loss=target_skill_binary_crossentropy,
            optimizer=optimizer,
            metrics=metrics,
            experimental_run_tf_function=False)

    def fit(self,
            dataset,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1):
        """Trains the model for a fixed number of epochs (iterations on a dataset).
        Arguments:
            dataset: A `tf.data` dataset. Should return a tuple
                of `(inputs, (skills, targets))`.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch)
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. The default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is exhausted.
            validation_steps: Only relevant if `validation_data` is provided.
                Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If
                'validation_steps' is None, validation
                will run until the `validation_data` dataset is exhausted.
            validation_freq: Only relevant if validation data is provided. Integer
                or `collections_abc.Container` instance (e.g. list, tuple, etc.).
                If an integer, specifies how many training epochs to run before a
                new validation run is performed, e.g. `validation_freq=2` runs
                validation every 2 epochs. If a Container, specifies the epochs on
                which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
                validation at the end of the 1st, 2nd, and 10th epochs.
        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        Raises:
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        return super(DKTModel, self).fit(x=dataset,
                                         epochs=epochs,
                                         verbose=verbose,
                                         callbacks=callbacks,
                                         validation_data=validation_data,
                                         shuffle=shuffle,
                                         initial_epoch=initial_epoch,
                                         steps_per_epoch=steps_per_epoch,
                                         validation_steps=validation_steps,
                                         validation_freq=validation_freq)

    def evaluate(self,
                 dataset,
                 verbose=1,
                 steps=None,
                 callbacks=None):
        """Returns the loss value & metrics values for the model in test mode.
        Computation is done in batches.
        Arguments:
            dataset: `tf.data` dataset. Should return a
            tuple of `(inputs, (skills, targets))`.
            verbose: 0 or 1. Verbosity mode.
                0 = silent, 1 = progress bar.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.
                If x is a `tf.data` dataset and `steps` is
                None, 'evaluate' will run until the dataset is exhausted.
                This argument is not supported with array inputs.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
                See [callbacks](/api_docs/python/tf/keras/callbacks).
        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        Raises:
            ValueError: in case of invalid arguments.
        """
        return super(DKTModel, self).evaluate(dataset,
                                              verbose=verbose,
                                              steps=steps,
                                              callbacks=callbacks)

    def evaluate_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")

    def fit_generator(self, *args, **kwargs):
        raise SyntaxError("Not supported")


def load_dataset(fn, batch_size=32, shuffle=True):
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
    nb_users = len(seq)

    # Step 5 - Get Tensorflow Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: seq,
        output_types=(tf.int32, tf.int32, tf.float32)
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=nb_users)

    # Step 6 - Pad sequences per batch
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padding_values=(-1, -1, 0.),
        padded_shapes=([None], [None], [None]),
        drop_remainder=True
    )

    # Step 7 - Encode categorical features and merge skills with labels to compute target loss.
    # More info: https://github.com/tensorflow/tensorflow/issues/32142
    features_depth = df['skill_with_answer'].max() + 1
    skill_depth = df['skill'].max() + 1

    dataset = dataset.map(
        lambda feat, skill, label: (
            tf.one_hot(feat, depth=features_depth),
            tf.concat(values=[
                tf.one_hot(skill, depth=skill_depth),
                tf.expand_dims(label, -1)
            ], axis=-1)
        )
    )

    length = nb_users // batch_size
    return dataset, length, features_depth, skill_depth


def split_dataset(dataset, total_size, test_fraction, val_fraction=None):
    def split(dataset, split_size):
        split_set = dataset.take(split_size)
        dataset = dataset.skip(split_size)
        return dataset, split_set

    assert 0 < test_fraction < 1
    assert val_fraction is None or 0 < val_fraction < 1

    test_size = int(test_fraction * total_size)
    train_set, test_set = split(dataset, test_size)

    val_set = None
    if val_fraction:
        val_size = int((total_size - test_size) * val_fraction)
        train_set, val_set = split(train_set, val_size)

    return train_set, test_set, val_set