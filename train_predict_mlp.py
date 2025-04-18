import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset, SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.utils import resample
import pickle
import os
import datetime
from tqdm import tqdm
import wandb

# --- Configs ---
USE_WANDB = False
MODEL_TYPE = 'bagging'  # single, bagging, adaboost
N_ESTIMATORS = 10
NUM_EPOCHS_PER_ESTIMATOR = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT_RATE = 0.4
EMBEDDING_DIM_BREED = 4
EMBEDDING_DIM_COLOR = 3
VALIDATION_SPLIT = 0.15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

PROCESSED_DIR = 'processed_data'
train_data_path = os.path.join(PROCESSED_DIR, 'proc_train_data.pkl')
mappings_path = os.path.join(PROCESSED_DIR, 'proc_mappings.pkl')

with open(train_data_path, 'rb') as f:
    train_data_dict = pickle.load(f)
with open(mappings_path, 'rb') as f:
    mappings = pickle.load(f)

X_df = train_data_dict['X']
y_series = train_data_dict['y']

# Get info from mappings
label_encoders = mappings['label_encoders']
vocab_sizes = mappings['vocab_sizes']
numerical_cols = mappings['numerical_cols']
final_columns = mappings['final_columns']
num_classes = len(y_series.unique())

# Identify column indices for the model
embed_col_indices = {
    'Breed': X_df.columns.get_loc('Breed_encoded'),
    'Color': X_df.columns.get_loc('Color_encoded')
}
# All other columns are treated as numerical (including OHE features)
# Exclude the embedding columns themselves
numerical_indices = [i for i, col in enumerate(X_df.columns) if '_encoded' not in col]

X_np = X_df.values.astype(np.float32)
y_np = y_series.values


# --- Create PyTorch Dataset ---
class MixedInputDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]  # For prediction


full_dataset = MixedInputDataset(X_np, y_np)

# --- Train/Validation Split ---
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=VALIDATION_SPLIT,
    stratify=y_np,  # Stratify by target class
    random_state=42
)

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)  # Larger batch for eval

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Calculate Class Weights for Weighted Loss
train_labels = y_np[train_indices]  # Get labels only from the training subset
class_counts = np.bincount(train_labels, minlength=num_classes)
total_train_samples = len(train_labels)

# Handle division by zero if a class is missing in the split
class_weights = np.array([total_train_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts])

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights_tensor.cpu().numpy()}")


# --- Define MLP Model with Embeddings ---
class MLPWithEmbeddings(nn.Module):
    def __init__(self, num_numerical_features, vocab_sizes, embedding_dims,
                 hidden_dims, output_dim, dropout_rate):
        super().__init__()
        self.embed_col_indices = embed_col_indices
        self.numerical_indices = numerical_indices

        # Embedding layers
        self.embedding_breed = nn.Embedding(vocab_sizes['Breed'], embedding_dims['Breed'])
        self.embedding_color = nn.Embedding(vocab_sizes['Color'], embedding_dims['Color'])
        total_embedding_dim = embedding_dims['Breed'] + embedding_dims['Color']

        input_dim = num_numerical_features + total_embedding_dim
        print("input dim:", input_dim, "num_numerical_featues:", num_numerical_features, "total_embedding_dim:", total_embedding_dim)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        numerical_input = x[:, self.numerical_indices].float()
        breed_input = x[:, self.embed_col_indices['Breed']].long()
        color_input = x[:, self.embed_col_indices['Color']].long()

        # Process embeddings
        embed_breed = self.embedding_breed(breed_input)
        embed_color = self.embedding_color(color_input)

        # Concatenate
        combined = torch.cat([numerical_input, embed_breed, embed_color], dim=1)

        out = self.dropout1(self.relu1(self.fc1(combined)))
        out = self.dropout2(self.relu2(self.fc2(out)))
        out = self.fc_out(out)
        return out


def train_one_epoch(model, loader, criterion, optimizer, device, sample_weights=None):
    model.train()
    running_loss = 0.0
    num_samples = 0

    # Decide loader based on sample_weights for AdaBoost
    if sample_weights is not None:
        # Create a sampler for the current weights
        # Ensure sample_weights is a CPU tensor/numpy array for sampler
        weights_tensor = torch.tensor(sample_weights, dtype=torch.float32).cpu()
        sampler = WeightedRandomSampler(weights_tensor, len(weights_tensor), replacement=True)
        current_loader = DataLoader(loader.dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        current_loader = loader


    for inputs, labels in current_loader:  # Use the potentially weighted loader
        inputs, labels = inputs.to(device), labels.to(device)
        num_samples += inputs.size(0)

        optimizer.zero_grad()
        outputs = model(inputs)  # Logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)  # Weighted by batch size

    return running_loss / len(loader.dataset)  # Average loss over all samples in the original subset


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds), balanced_accuracy_score(all_labels, all_preds)


# --- Model Initialization ---
num_numerical = len(numerical_indices)
model_args = {
    'num_numerical_features': num_numerical,
    'vocab_sizes': vocab_sizes,
    'embedding_dims': {'Breed': EMBEDDING_DIM_BREED, 'Color': EMBEDDING_DIM_COLOR},
    'hidden_dims': [HIDDEN_DIM_1, HIDDEN_DIM_2],
    'output_dim': num_classes,
    'dropout_rate': DROPOUT_RATE
}

if USE_WANDB:
    cur_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{MODEL_TYPE}_MLP_{cur_date_time}"
    wandb.init(entity='evan-kuo-edu', name=run_name, project="ML_Course_Project_Animal_Outcomes", config={ # CHANGE ENTITY
                                                                                                          "model_type": MODEL_TYPE,
                                                                                                          "n_estimators": N_ESTIMATORS if MODEL_TYPE != 'single' else 1,
                                                                                                          "epochs_per_estimator": NUM_EPOCHS_PER_ESTIMATOR,
                                                                                                          "batch_size": BATCH_SIZE,
                                                                                                          "learning_rate": LEARNING_RATE,
                                                                                                          "dropout": DROPOUT_RATE,
                                                                                                          "hidden_dims": [HIDDEN_DIM_1, HIDDEN_DIM_2],
                                                                                                          "embed_breed": EMBEDDING_DIM_BREED,
                                                                                                          "embed_color": EMBEDDING_DIM_COLOR,
                                                                                                          "validation_split": VALIDATION_SPLIT
                                                                                                          })

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # use class weights

if MODEL_TYPE == 'single':
    print("--- Training Single MLP ---")
    model = MLPWithEmbeddings(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_val_balanced_acc = -1
    best_model_state = None

    for epoch in range(NUM_EPOCHS_PER_ESTIMATOR * N_ESTIMATORS):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_balanced_acc = evaluate_model(model, val_loader, device)
        print(f'Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_balanced_acc:.4f}')

        if USE_WANDB:
            wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "val_accuracy": val_acc, "val_balanced_accuracy": val_balanced_acc})

        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_model_state = model.state_dict()
            print(f"  * New best validation balanced accuracy: {best_val_balanced_acc:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    final_val_acc, final_val_bal_acc = evaluate_model(model, val_loader, device)
    print(f"\nFinal Single Model Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Single Model Validation Balanced Accuracy: {final_val_bal_acc:.4f}")
    if USE_WANDB:
        wandb.log({"final_val_accuracy": final_val_acc, "final_val_balanced_accuracy": final_val_bal_acc})
    torch.save(model.state_dict(), f"single_mlp_best_{cur_date_time}.pt")


elif MODEL_TYPE == 'bagging':
    print(f"--- Training Bagging MLP Ensemble ({N_ESTIMATORS} estimators) ---")
    estimators = []
    n_train_samples = len(train_dataset)

    for i in range(N_ESTIMATORS):
        print(f"\nTraining Estimator {i + 1}/{N_ESTIMATORS}...")
        # Create bootstrap sample indices
        bootstrap_indices = resample(train_indices, n_samples=n_train_samples, replace=True, random_state=i)
        bootstrap_sampler = SubsetRandomSampler(bootstrap_indices)
        train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=bootstrap_sampler)

        model = MLPWithEmbeddings(**model_args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS_PER_ESTIMATOR):
            epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # Optional: evaluate on validation set per epoch if desired
            if (epoch + 1) % 5 == 0: # Evaluate less frequently
                val_acc, val_balanced_acc = evaluate_model(model, val_loader, device)
                print(f' Estimator {i+1}, Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}, Val Bal Acc: {val_balanced_acc:.4f}')
                if USE_WANDB:
                    wandb.log({f"estimator_{i + 1}_epoch": epoch + 1, f"estimator_{i+1}_loss": epoch_loss, f"estimator_{i+1}_val_balanced_acc": val_balanced_acc})

        estimators.append(model.eval())

    # Evaluate the ensemble
    print("\n--- Evaluating Bagging Ensemble ---")
    all_probs = []
    all_labels_val = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            batch_probs = []
            for model in estimators:
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_probs.append(probs.cpu().numpy())
            # Average probabilities across estimators for this batch
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs.append(avg_probs)
            all_labels_val.extend(labels.cpu().numpy())

    final_probs = np.concatenate(all_probs, axis=0)
    final_preds = np.argmax(final_probs, axis=1)

    ensemble_val_acc = accuracy_score(all_labels_val, final_preds)
    ensemble_val_bal_acc = balanced_accuracy_score(all_labels_val, final_preds)

    print(f"Final Ensemble Validation Accuracy: {ensemble_val_acc:.4f}")
    print(f"Final Ensemble Validation Balanced Accuracy: {ensemble_val_bal_acc:.4f}")
    if USE_WANDB:
        wandb.log({"final_ensemble_val_accuracy": ensemble_val_acc, "final_ensemble_val_balanced_accuracy": ensemble_val_bal_acc})
    # Save the list of estimators if needed
    torch.save([est.state_dict() for est in estimators], f"bagging_mlp_{cur_date_time}.pt")


elif MODEL_TYPE == 'adaboost':
    print(f"--- Training AdaBoost (Simulated) MLP Ensemble ({N_ESTIMATORS} estimators) ---")
    estimators = []
    alphas = []
    n_train_samples = len(train_dataset)
    sample_weights = np.ones(n_train_samples) / n_train_samples

    # Need labels for the training subset to calculate errors/update weights
    train_subset_labels = y_np[train_indices]

    # Create a loader for the *entire* training subset for error calculation
    # No shuffling needed here as we use indices directly
    full_train_subset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    for i in range(N_ESTIMATORS):
        print(f"\nTraining Estimator {i + 1}/{N_ESTIMATORS}...")
        model = MLPWithEmbeddings(**model_args).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Train model using weighted sampling
        for epoch in range(NUM_EPOCHS_PER_ESTIMATOR):
            # Pass the numpy weights, train_one_epoch will handle sampler creation
            epoch_loss = train_one_epoch(model, full_train_subset_loader, criterion, optimizer, device, sample_weights=sample_weights)
            if (epoch + 1) % 5 == 0:
                print(f' Estimator {i+1}, Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}')
                if USE_WANDB:
                    wandb.log({f"estimator_{i+1}_epoch": epoch + 1, f"estimator_{i+1}_loss": epoch_loss})

        # Calculate weighted error on the training subset
        model.eval()
        estimator_preds = []
        with torch.no_grad():
            for inputs, _ in full_train_subset_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                estimator_preds.extend(predicted.cpu().numpy())

        estimator_preds = np.array(estimator_preds)
        misclassified_mask = estimator_preds != train_subset_labels
        weighted_error = np.sum(sample_weights[misclassified_mask])

        # Avoid division by zero or log(0)
        epsilon = max(weighted_error, 1e-10)
        epsilon = min(epsilon, 1.0 - 1e-10) # Ensure error is not exactly 1

        # Calculate estimator weight (alpha) - SAMME.R logic is complex, use AdaBoost.M1 logic
        alpha = np.log((1.0 - epsilon) / epsilon) + np.log(num_classes - 1)
        print(f" Estimator {i+1}: Weighted Error={epsilon:.4f}, Alpha={alpha:.4f}")

        # Update sample weights
        # Increase weight for misclassified samples, decrease for correct ones
        sample_weights *= np.exp(alpha * misclassified_mask)
        # Normalize weights
        sample_weights /= np.sum(sample_weights)

        estimators.append(model.eval())
        alphas.append(alpha)

        if epsilon >= (1.0 - 1.0 / num_classes):
            print(f" Estimator {i+1} is worse than random guessing. Stopping early.")
            if i == 0:
                print("Warning: First estimator is poor. Ensemble may not perform well.")
            else:
                # Remove the last estimator and alpha
                estimators.pop()
                alphas.pop()
            break  # Stop boosting if error is too high

    print("\n--- Evaluating AdaBoost Ensemble ---")
    all_weighted_preds = np.zeros((len(val_dataset), num_classes))
    all_labels_val = []

    with torch.no_grad():
        # Get labels from val_loader first
        for _, labels in val_loader:
            all_labels_val.extend(labels.cpu().numpy())

        # Get predictions batch by batch and accumulate weighted votes
        current_idx = 0
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            batch_size_current = inputs.size(0)
            for model, alpha in zip(estimators, alphas):
                outputs = model(inputs)
                _, predicted_class = torch.max(outputs, 1)
                # Add alpha to the predicted class vote for each sample in the batch
                for j in range(batch_size_current):
                    all_weighted_preds[current_idx + j, predicted_class[j].item()] += alpha
            current_idx += batch_size_current

    final_preds = np.argmax(all_weighted_preds, axis=1)

    ensemble_val_acc = accuracy_score(all_labels_val, final_preds)
    ensemble_val_bal_acc = balanced_accuracy_score(all_labels_val, final_preds)

    print(f"Final AdaBoost Ensemble Validation Accuracy: {ensemble_val_acc:.4f}")
    print(f"Final AdaBoost Ensemble Validation Balanced Accuracy: {ensemble_val_bal_acc:.4f}")

    if USE_WANDB:
        wandb.log({"final_ensemble_val_accuracy": ensemble_val_acc, "final_ensemble_val_balanced_accuracy": ensemble_val_bal_acc})
    save_obj = {'estimators': [est.state_dict() for est in estimators], 'alphas': alphas}
    torch.save(save_obj, f"adaboost_mlp_{cur_date_time}.pt")


else:
    print(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

if USE_WANDB:
    wandb.finish()

print("Training finished.")


# ========================================
# ===== PREDICTION ON TEST DATA =========
# ========================================
print("\n--- Generating Predictions on Test Data ---")

# --- Load Test Data ---
test_data_path = os.path.join(PROCESSED_DIR, 'proc_test_data.pkl')
if not os.path.exists(test_data_path):
    print(f"ERROR: Processed test data not found at {test_data_path}")
    print("Please run datapreprocessing.py with PROCESS_TRAIN_DATA = False")
else:
    with open(test_data_path, 'rb') as f:
        test_data_dict = pickle.load(f)

    X_test_df = test_data_dict['X_test']
    test_ids = X_test_df['Id']  # Keep track of original IDs
    X_test_df = X_test_df.drop(columns=['Id'])  # Drop Id before converting to numpy

    # Ensure columns match training data (should be handled by preprocessing)
    if not all(X_test_df.columns == X_df.columns):  # X_df is from training data loading
        print("WARNING: Test columns do not perfectly match training columns AFTER loading!")
        # Attempt to reorder/add/remove based on training columns if needed
        train_cols_no_id = [col for col in final_columns if col != 'Id']
        X_test_df = X_test_df.reindex(columns=train_cols_no_id, fill_value=0)
        print("Columns realigned.")

    X_test_np = X_test_df.values.astype(np.float32)
    test_dataset_obj = MixedInputDataset(X_test_np)  # No labels for test set
    test_loader = DataLoader(test_dataset_obj, batch_size=BATCH_SIZE * 2, shuffle=False)

    final_test_preds = []

    if MODEL_TYPE == 'single':
        if 'model' not in locals() or best_model_state is None:
            print("ERROR: Single model not found or not trained properly with best state saved.")
        else:
            print("Using trained single model (best state based on validation).")
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                for inputs in tqdm(test_loader, desc="Predicting Test (Single)"):
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    final_test_preds.extend(predicted.cpu().numpy())

    elif MODEL_TYPE == 'bagging':
        if 'estimators' not in locals() or not estimators:
            print("ERROR: Bagging estimators not found or not trained.")
        else:
            print(f"Using trained bagging ensemble ({len(estimators)} estimators).")
            all_test_probs = []
            with torch.no_grad():
                for inputs in tqdm(test_loader, desc="Predicting Test (Bagging)"):
                    inputs = inputs.to(device)
                    batch_probs = []
                    for est in estimators:
                        est.eval()
                        outputs = est(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        batch_probs.append(probs.cpu().numpy())
                    avg_probs = np.mean(batch_probs, axis=0)
                    all_test_probs.append(avg_probs)
            final_probs = np.concatenate(all_test_probs, axis=0)
            final_test_preds = np.argmax(final_probs, axis=1)

    elif MODEL_TYPE == 'adaboost':
        if 'estimators' not in locals() or not estimators or 'alphas' not in locals() or not alphas:
            print("ERROR: AdaBoost estimators/alphas not found or not trained.")
        else:
            print(f"Using trained AdaBoost ensemble ({len(estimators)} estimators).")
            num_test_samples = len(test_dataset_obj)
            all_weighted_preds_test = np.zeros((num_test_samples, num_classes))
            current_idx = 0
            with torch.no_grad():
                for inputs in tqdm(test_loader, desc="Predicting Test (AdaBoost)"):
                    inputs = inputs.to(device)
                    batch_size_current = inputs.size(0)
                    for model, alpha in zip(estimators, alphas):
                        model.eval()
                        outputs = model(inputs)
                        _, predicted_class = torch.max(outputs, 1)
                        for j in range(batch_size_current):
                            all_weighted_preds_test[current_idx + j, predicted_class[j].item()] += alpha
                    current_idx += batch_size_current
            final_test_preds = np.argmax(all_weighted_preds_test, axis=1)

    else:
        print(f"Prediction logic not implemented for MODEL_TYPE: {MODEL_TYPE}")

    # --- Format Submission ---
    if len(final_test_preds) > 0:
        outcome_mapping = {
            'Return to Owner': 0, 'Transfer': 1, 'Adoption': 2,
            'Died': 3, 'Euthanasia': 4
        }
        reverse_outcome_mapping = {v: k for k, v in outcome_mapping.items()}

        predicted_labels = [reverse_outcome_mapping[pred] for pred in final_test_preds]

        submission_df = pd.DataFrame({'Id': test_ids, 'Outcome Type': predicted_labels})

        os.makedirs('results', exist_ok=True)
        submission_filename = os.path.join('results', f'submission_{MODEL_TYPE}_{cur_date_time if "cur_date_time" in locals() else "mlp"}.csv')
        submission_df.to_csv(submission_filename, index=False)
        print(f"\nSubmission file saved to: {submission_filename}")
    else:
        print("\nNo predictions generated. Submission file not created.")
