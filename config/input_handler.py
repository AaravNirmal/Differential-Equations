# config/input_handler.py
import json

def load_config(path="config/default_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def user_override(config):
    print("🔧 CONFIGURATION OVERRIDE (press enter to skip)")
    try:
        dim = input(f"Dimension [{config['dimension']}]: ")
        if dim: config["dimension"] = int(dim)

        initial = input(f"Initial state (comma-separated) [{config['initial_state']}]: ")
        if initial:
            config["initial_state"] = list(map(float, initial.split(",")))

        T = input(f"Simulation time T [{config['T']}]: ")
        if T: config["T"] = float(T)

        dt = input(f"Time step dt [{config['dt']}]: ")
        if dt: config["dt"] = float(dt)

        eqs = input("Enter field expressions (comma-separated, optional): ")
        if eqs:
            config["field_exprs"] = [e.strip() for e in eqs.split(",")]
    except Exception as e:
        print(f"[!] Error reading input: {e}")
    return config

def save_config(config, path="config/last_used_config.json"):
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
