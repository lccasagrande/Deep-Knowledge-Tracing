from Utils import *
import argparse
from StudentModel import Model, DataGenerator


def run(args):
    # Load dataset
    dataset, num_skills = read_file(args.dataset)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(dataset,
                                                                   validation_rate=args.validation_rate,
                                                                   testing_rate=args.testing_rate)

    if(args.verbose):
        print("==== Data Summary ====")
        print("Training data: %d" % len(X_train))
        print("Validation data: %d" % len(X_val))
        print("Testing data: %d" % len(X_test))
        print("Number of skills: %d" % num_skills)
        print("======================")

    # Create generators
    train_gen = DataGenerator(X_train, y_train, num_skills, args.batch_size)
    val_gen = DataGenerator(X_val, y_val, num_skills, args.batch_size)
    test_gen = DataGenerator(X_test, y_test, num_skills, args.batch_size)

    # Create model
    student_model = Model(num_skills=train_gen.num_skills,
                          num_features=train_gen.feature_dim,
                          optimizer=args.optimizer,
                          hidden_units=args.lstm_units,
                          batch_size=args.batch_size,
                          dropout_rate=args.dropout_rate)

    # Train
    student_model.fit(train_gen,
                      epochs=args.epochs,
                      val_gen=val_gen,
                      verbose=args.verbose,
                      file=args.best_model_file,
                      log_file=args.train_log)

    # Load the model with the best Validation Loss
    student_model.load_weights(args.best_model_file)

    # Test
    student_model.evaluate(test_gen, metrics=['auc','acc','pre'], verbose=args.verbose, log_file=args.eval_log)

    if args.verbose:
        print("======= DONE =======")

    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DKT Model")

    parser.add_argument("--dataset", type=str, default="data/ASSISTments_skill_builder_data.csv", help="Dataset to load.")
    parser.add_argument("--best_model_file", type=str, default="saved_models/ASSISTments.best.model.weights.hdf5", help="The file to save the model.")
    parser.add_argument("--train_log", type=str, default="logs/dktmodel.train.log", help="The file to save training log.")
    parser.add_argument("--eval_log", type=str, default="logs/dktmodel.eval.log", help="The file to save evaluation log.")
    parser.add_argument("--optimizer", type=str, default="adagrad", help="The optimizer to use.")
    parser.add_argument("--lstm_units", type=int, default=250, help="The number of LSTM hidden units.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to run.")
    parser.add_argument("--dropout_rate", type=float, default=0.6, help="The dropout probability.")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--validation_rate", type=float, default=0.2, help="Portion of training data to be used for validation.")
    parser.add_argument("--testing_rate", type=float, default=0.2, help="Portion of data to be used for testing")

    FLAGS, _ = parser.parse_known_args()

    run(FLAGS)


