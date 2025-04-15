import os
import pandas as pd
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inverse mapping: from class index to outcome string.
inv_outcome_mapping = {
    0: 'Return to Owner',
    1: 'Transfer',
    2: 'Adoption',
    3: 'Died',
    4: 'Euthanasia'
}

# ---------------------------
# Step 1: Load the Test Data
# ---------------------------
test_data_path = os.path.join('data', 'processed_test_data.pkl')
test_df = pd.read_pickle(test_data_path)
print("Test dataframe loaded.")

# Extract the Id column for the final submission and drop it from the features.
ids = test_df['Id']
test_features = test_df.drop(columns=['Id', 'Color']).to_numpy().astype(np.float32)

# Convert test features to a torch tensor.
test_tensor = torch.tensor(test_features).to(device)

# ---------------------------
# Step 2: Load the Saved Model
# ---------------------------
# Update the model path as needed (this should match how you saved it).
# model_save_path = os.path.join('results', 'MLP_2025-04-05_23-25-55.pt')
# checkpoint = torch.load(model_save_path, map_location=device)

# The checkpoint was saved as a dictionary with "model" and "states".
# We retrieve the model and load its state dictionary.
# model = checkpoint["model"]
# model.load_state_dict(checkpoint["states"])

model_save_path = os.path.join('results', 'MLP_2025-04-06_12-59-20.pt')
model = torch.jit.load(model_save_path)

model.to(device)
model.eval()  # Set the model to evaluation mode.
print("Model loaded and set to evaluation mode.")

# ---------------------------
# Step 3: Predict the Outcomes
# ---------------------------
with torch.no_grad():
    outputs = model(test_tensor)
    # Get the predicted class for each record (the index of the max logit).
    _, predicted_indices = torch.max(outputs, 1)

# Convert predictions from tensor to a list of outcome strings.
predictions = [inv_outcome_mapping[int(idx)] for idx in predicted_indices]
print("Predictions completed.")

# ---------------------------
# Step 4: Create and Save the Submission File
# ---------------------------
submission_df = pd.DataFrame({
    'Id': ids,
    'Outcome Type': predictions
})

# Save to CSV without the DataFrame index.
submission_csv_path = 'submission.csv'
submission_df.to_csv(submission_csv_path, index=False)
print(f"Submission saved to {submission_csv_path}.")