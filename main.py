# main.py

from src.data_loader import load_data, get_feature_cols
from src.splitter import make_splits
from src.visualize import inspect_data, plot_samples, plot_augmentations
from src.train_svm import cross_validate, train_final
from src.evaluate import evaluate_on_test, print_confusion_matrix


def main():
    # 1. Load data
    df = load_data()
    inspect_data(df)
    feature_cols = get_feature_cols(df)

    # 2. Visualize samples & augmentations
    plot_samples(df)
    plot_augmentations(df)

    # 3. Split
    X_train_full, y_train_full, g_train_full, X_test, y_test, split_iter = \
        make_splits(df, feature_cols)

    # 4. Cross-validate
    cross_validate(X_train_full, y_train_full, split_iter)

    # 5. Train final model & evaluate on test
    final_pipe = train_final(X_train_full, y_train_full)
    y_pred = evaluate_on_test(final_pipe, X_test, y_test)

    # 6. Confusion matrices
    print_confusion_matrix(y_test, y_pred, normalize=False)
    print_confusion_matrix(y_test, y_pred, normalize=True)


if __name__ == "__main__":
    main()