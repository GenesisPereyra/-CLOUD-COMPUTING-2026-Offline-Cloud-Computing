import os
import argparse
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from TheModel import create_model


def load_local_models(models_dir):
    model_files = [
        file for file in os.listdir(models_dir)
        if file.endswith(".keras")
    ]

    model_files.sort()

    models = []

    for file in model_files:
        path = os.path.join(models_dir, file)
        model = tf.keras.models.load_model(path)
        models.append(model)
        print(f"Modelo cargado: {file}")

    return models, model_files


def fedavg(models):
    weights = [model.get_weights() for model in models]
    averaged_weights = []

    for layer_weights in zip(*weights):
        averaged_weights.append(np.mean(layer_weights, axis=0))

    return averaged_weights


def weighted_fedavg(models, client_weights):
    weights = [model.get_weights() for model in models]
    client_weights = np.array(client_weights)
    client_weights = client_weights / np.sum(client_weights)

    averaged_weights = []

    for layer_weights in zip(*weights):
        layer_sum = np.zeros_like(layer_weights[0])

        for client_id, weights_value in enumerate(layer_weights):
            layer_sum += client_weights[client_id] * weights_value

        averaged_weights.append(layer_sum)

    return averaged_weights


def trimmed_mean(models, trim_ratio=0.1):
    weights = [model.get_weights() for model in models]
    aggregated_weights = []

    for layer_weights in zip(*weights):
        stacked = np.stack(layer_weights, axis=0)

        n_clients = stacked.shape[0]
        trim_count = int(trim_ratio * n_clients)

        sorted_weights = np.sort(stacked, axis=0)

        if trim_count > 0 and 2 * trim_count < n_clients:
            trimmed = sorted_weights[trim_count:-trim_count]
        else:
            trimmed = sorted_weights

        aggregated_layer = np.mean(trimmed, axis=0)
        aggregated_weights.append(aggregated_layer)

    return aggregated_weights


def evaluate_global_model(model, x_test, y_test, method_name, output_dir):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred)

    print("\n===================================")
    print(f"Resultados método: {method_name}")
    print("Loss:", loss)
    print("Accuracy:", accuracy)
    print(report)

    with open(os.path.join(output_dir, f"{method_name}_classification_report.txt"), "w") as f:
        f.write(f"Method: {method_name}\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write(report)

    return loss, accuracy


def main(models_dir, test_data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    local_models, model_files = load_local_models(models_dir)

    if len(local_models) == 0:
        raise ValueError("No se encontraron modelos locales en la carpeta indicada.")

    test_data = np.load(test_data_path)
    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    methods = {
        "FedAvg": fedavg(local_models),
        "Weighted_FedAvg": weighted_fedavg(
    local_models,
    client_weights=[12005, 12002, 12000, 11997, 11996]
    ),
        "Trimmed_Mean": trimmed_mean(local_models, trim_ratio=0.2)
    }

    summary = []

    for method_name, global_weights in methods.items():
        global_model = create_model()
        global_model.set_weights(global_weights)

        loss, accuracy = evaluate_global_model(
            global_model,
            x_test,
            y_test,
            method_name,
            output_dir
        )

        global_model.save(os.path.join(output_dir, f"{method_name}_global_model.keras"))

        summary.append((method_name, loss, accuracy))

    with open(os.path.join(output_dir, "global_summary.txt"), "w") as f:
        for method_name, loss, accuracy in summary:
            f.write(f"{method_name}: loss={loss}, accuracy={accuracy}\n")

    print("\nResumen global:")
    for method_name, loss, accuracy in summary:
        print(f"{method_name}: loss={loss:.4f}, accuracy={accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models_dir",
        type=str,
        default="local_models"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default="../data_private/mnist_test.npz"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="global_models"
    )

    args = parser.parse_args()

    main(args.models_dir, args.test_data, args.output_dir)