from config.input_handler import load_config, user_override, save_config
from field.vector_field import make_field_from_expr
from integrator.integrator import simulate
from reducer.reducer import reduce_to_2d
from visualizer.plotter import plot_2d
from visualizer.animator import animate
from utils.save_results import save_output
import numpy as np


def main():
    # Load and optionally override config
    config = load_config()
    config = user_override(config)
    save_config(config)

    # Build vector field from expressions
    field = make_field_from_expr(config["field_exprs"])

    # Simulate particle trajectory
    X_real = simulate(field, config["initial_state"], config["T"], config["dt"])

    # Reduce high-D trajectory to 2D for visualization
    X_real_2D = reduce_to_2d(X_real)

    # Save results: .npy, .csv, plot
    save_output(X_real, X_real_2D, tag="real")

    # Visualize animated 2D trajectory
    animate(X_real_2D)

if __name__ == "__main__":
    main()
