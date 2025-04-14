from config.input_handler import load_config, user_override, save_config
from field.field import make_field_from_expr
from integrator.integrator import simulate
from ml.ml_model import train_model, predict_trajectory
from visualizer.animator import animate
from sklearn.decomposition import PCA
import numpy as np

def prepare_ml_data(X):
    return X[:-1], X[1:]

def main():
    config = load_config()
    config = user_override(config)

    field = make_field_from_expr(config["field_exprs"])
    X_real = simulate(field, config["initial_state"], config["T"], config["dt"])

    X_train, Y_train = prepare_ml_data(X_real)
    model = train_model(X_train, Y_train, config["dimension"],
                        config["ml"]["epochs"], config["ml"]["learning_rate"])

    X_pred = predict_trajectory(model, X_real[0], len(X_real) - 1)

    pca = PCA(n_components=2)
    X_real_2D = pca.fit_transform(X_real)
    X_pred_2D = pca.transform(X_pred)

    animate(X_real_2D, X_pred_2D)

if __name__ == "__main__":
    main()
