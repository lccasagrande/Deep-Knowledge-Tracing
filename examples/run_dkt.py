import argparse

from deepkt import deepkt, data_util


def run(args):
    dataset, length, nb_features, nb_skills = data_util.load_dataset(fn=args.f,
                                                                     batch_size=args.batch_size,
                                                                     shuffle=True)

    train_set, test_set, val_set = data_util.split_dataset(dataset=dataset,
                                                           total_size=length,
                                                           test_fraction=args.test_split,
                                                           val_fraction=args.val_split)

    model = deepkt.DKTModel(nb_features=nb_features,
                            nb_skills=nb_skills,
                            hidden_units=args.hidden_units,
                            dropout_rate=args.dropout_rate)

    model.compile(optimizer='rmsprop')

    print("\n[----- TRAINING ------]")

    model.fit(dataset=train_set,
              epochs=args.epochs,
              verbose=args.v,
              validation_data=val_set)

    print("\n[--- TRAINING DONE ---]")

    print("[----- TESTING  ------]")

    model.evaluate(dataset=test_set,
                   verbose=args.v)

    print("\n[--- TESTING DONE  ---]")


def parse_args():
    parser = argparse.ArgumentParser(prog="DeepKT Example")

    parser.add_argument("-f",
                        type=str,
                        default="data/ASSISTments_skill_builder_data.csv",
                        help="the path to the data")

    parser.add_argument("-v",
                        type=int,
                        default=1,
                        help="verbosity mode [0, 1, 2].")

    model_group = parser.add_argument_group(title="Model arguments.")
    model_group.add_argument("--dropout_rate",
                             type=float,
                             default=.2,
                             help="fraction of the units to drop.")
    model_group.add_argument("--hidden_units",
                             type=int,
                             default=100,
                             help="number of units of the LSTM layer.")

    train_group = parser.add_argument_group(title="Training arguments.")
    train_group.add_argument("--batch_size",
                             type=int,
                             default=16,
                             help="number of elements to combine in a single batch.")
    train_group.add_argument("--epochs",
                             type=int,
                             default=3,
                             help="number of epochs to train.")

    train_group.add_argument("--test_split",
                             type=float,
                             default=.2,
                             help="fraction of data to be used for testing (0, 1).")

    train_group.add_argument("--val_split",
                             type=float,
                             default=.2,
                             help="fraction of data to be used for validation (0, 1).")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
