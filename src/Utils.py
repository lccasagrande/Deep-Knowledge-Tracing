import pandas as pd
import random


def split_dataset(data, validation_rate, testing_rate, shuffle=True):
    def split(dt):
        return [[value[0] for value in seq] for seq in dt], [[value[1] for value in seq] for seq in dt]

    seqs = data
    if shuffle:
        random.shuffle(seqs)

    # Get testing data
    test_idx = random.sample(range(0, len(seqs)-1), int(len(seqs) * testing_rate))
    X_test, y_test = split([value for idx, value in enumerate(seqs) if idx in test_idx])
    seqs = [value for idx, value in enumerate(seqs) if idx not in test_idx]

    # Get validation data
    val_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * validation_rate))
    X_val, y_val = split([value for idx, value in enumerate(seqs) if idx in val_idx])

    # Get training data
    X_train, y_train = split([value for idx, value in enumerate(seqs) if idx not in val_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test


def read_file(dataset_path):
    data = pd.read_csv(dataset_path, dtype={'skill_name': str})

    # Step 1 - Remove problems without a skill_id
    data.dropna(subset=['skill_id'], inplace=True)

    # Step 2 - Convert to sequence by student id
    students_seq = data.groupby("user_id", as_index=True)["skill_id", "correct"].apply(lambda x: x.values.tolist()).tolist()

    # Step 3 - Rearrange the skill_id
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0

    for seq_idx, seq in enumerate(students_seq):
        for (skill, answer) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            if skill not in skill_ids:
                skill_ids[skill] = num_skill
                num_skill += 1

            seqs_by_student[seq_idx].append((skill_ids[skill], answer))

    return list(seqs_by_student.values()), num_skill
