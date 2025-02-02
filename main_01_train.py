"""
main_01_train.py

Training script for Blog Post 1 "Introduction to Attention Mechanisms with a Toy Dataset".

This updated version imports additional configuration parameters from config.py and uses them
to instantiate models (e.g., setting hidden dimension and number of heads) and to set training hyperparameters.
"""

import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from data.shapes_1d import generate_shapes
from train.train import train_model, evaluate_model
from models.basic import FCNModel, CNNModel, LSTMModel
from models.attention import (
    StandardAttentionModel,
    AttentionModelWithPE,
    MultiHeadAttentionModel,
)
from utils.metrics import count_parameters, count_flops
from config import (
    DEVICE,
    NUM_EPOCHS,
    BATCH_SIZE,
    NUM_TRAIN_SAMPLES,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    HIDDEN_DIM,
    NUM_HEADS,
)
from utils.environment import set_seed

# Set seed and device.
set_seed(42)
device = DEVICE
print(f"Using device: {device}")

# Create results folder.
results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)


def main():
    # --- Define models for training using configuration parameters ---
    models_dict = {
        "FCN": FCNModel(seq_length=SEQUENCE_LENGTH),
        "CNN": CNNModel(),
        "LSTM": LSTMModel(hidden_size=HIDDEN_DIM),
        "Attention": StandardAttentionModel(hidden_channels=HIDDEN_DIM),
        "Attention_PE": AttentionModelWithPE(hidden_channels=HIDDEN_DIM),
        "MultiHeadAttention": MultiHeadAttentionModel(
            n_heads=NUM_HEADS, hidden_channels=HIDDEN_DIM
        ),
    }

    # --- Compute Model Metrics (Parameters & FLOPs) ---
    model_metrics = []
    raw_dummy_input, _ = generate_shapes()
    base_dummy_input = torch.tensor(raw_dummy_input, dtype=torch.float32).to(device)

    for name, model in models_dict.items():
        model.to(device)
        param_count = count_parameters(model)
        if name == "FCN":
            model_dummy = base_dummy_input.unsqueeze(0)  # (1, seq_len)
        else:
            model_dummy = base_dummy_input.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len)
        flops_count = count_flops(model, model_dummy)
        model_metrics.append(
            {"Model": name, "Parameters": param_count, "FLOPs": flops_count}
        )

    metrics_df = pd.DataFrame(model_metrics)
    print("\n=== Model Metrics (Parameters & FLOPs) ===")
    print(metrics_df)

    # --- Train Each Model ---
    trained_results = {}
    loss_history = []

    for name, model in models_dict.items():
        print(f"\n--- Training model: {name} ---")
        losses, train_time = train_model(
            model=model,
            device=device,
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,  # Pass LEARNING_RATE from config (if supported)
        )
        best_train_loss = min(losses) if losses else float("inf")
        trained_results[name] = {
            "losses": losses,
            "train_time": train_time,
            "best_train_loss": best_train_loss,
            "model": model,
        }
        for idx, loss_val in enumerate(losses, start=1):
            loss_history.append({"Model": name, "Epoch": idx, "Loss": loss_val})

        weights_path = results_folder / f"blog_01_model_{name}_best_weights.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"Best weights for {name} saved to {weights_path.resolve()}")

    # --- Validation on a full batch per Model ---
    validation_results = {}
    print("\n=== Validation on full epoch per model ===")
    validation_data = []
    print("Generating validation dataset...")
    for _ in range(NUM_TRAIN_SAMPLES):
        input_seq, target_seq = generate_shapes()
        validation_data.append((input_seq, target_seq))

    for name, info in trained_results.items():
        inference_times = []
        validation_losses = []
        model = info["model"]
        model.eval()

        with torch.no_grad():
            for input_seq, target_seq in tqdm(
                validation_data, desc=f"Validating {name}", leave=False
            ):
                pred, inf_time = evaluate_model(model, input_seq, device=device)
                if not torch.is_tensor(target_seq):
                    target_tensor = torch.tensor(
                        target_seq, dtype=torch.float, device=device
                    )
                else:
                    target_tensor = target_seq.to(device)
                if not torch.is_tensor(pred):
                    pred = torch.tensor(pred, dtype=torch.float, device=device)
                loss_val = F.mse_loss(pred.cpu(), target_tensor.cpu()).item()
                inference_times.append(inf_time)
                validation_losses.append(loss_val)

        avg_inf_time = sum(inference_times) / len(inference_times)
        avg_val_loss = sum(validation_losses) / len(validation_losses)
        std_inf_time = torch.tensor(inference_times).std().item()
        std_val_loss = torch.tensor(validation_losses).std().item()

        validation_results[name] = {
            "avg_inference_time": avg_inf_time,
            "std_inference_time": std_inf_time,
            "avg_val_loss": avg_val_loss,
            "std_val_loss": std_val_loss,
        }
        print(
            f"Model {name}:\n"
            f"  Avg Inference Time: {avg_inf_time * 1000:.4f} ± {std_inf_time * 1000:.4f} ms\n"
            f"  Avg Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}"
        )

    # --- Combined Training & Validation Results ---
    combined_results = []
    for name, info in trained_results.items():
        val_info = validation_results[name]
        model_metric = next(m for m in model_metrics if m["Model"] == name)
        combined_results.append(
            {
                "Model": name,
                "Parameters": model_metric["Parameters"],
                "FLOPs": model_metric["FLOPs"],
                "Best Training Loss": info["best_train_loss"],
                "Training Time per Epoch (sec)": info["train_time"],
                "Average Validation Loss": val_info["avg_val_loss"],
                "Validation Loss Std": val_info["std_val_loss"],
                "Average Inference Time (ms)": val_info["avg_inference_time"] * 1000,
                "Inference Time Std (ms)": val_info["std_inference_time"] * 1000,
            }
        )
    combined_df = pd.DataFrame(combined_results)
    combined_csv_path = results_folder / "blog_01_table_combined_results.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print("\n=== Combined Training and Validation Results ===")
    print(combined_df)
    print(f"Combined results saved to {combined_csv_path.resolve()}")

    loss_history_df = pd.DataFrame(loss_history)
    loss_history_pivot = loss_history_df.pivot(
        index="Epoch", columns="Model", values="Loss"
    )
    loss_history_csv_path = results_folder / "blog_01_table_loss_history.csv"
    loss_history_pivot.to_csv(loss_history_csv_path)
    print("\nLoss history saved to", loss_history_csv_path.resolve())


if __name__ == "__main__":
    main()
