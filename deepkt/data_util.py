import pandas as pd
import tensorflow as tf


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


def get_target(y_true, y_pred):
    # Get skills and labels from y_true
    skills, y_true = tf.split(y_true, num_or_size_splits=[-1, 1], axis=-1)

    # Get predictions for each skill
    y_pred = tf.reduce_sum(y_pred * skills, axis=-1, keepdims=True)

    return y_true, y_pred
