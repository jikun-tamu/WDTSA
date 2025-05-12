import torch
import numpy as np
import pandas as pd
import random
import copy
import os
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split

from main import WideAndDeepClassifier, get_routed_columns, prepare_model_inputs, split, FEATURE_GROUPS, extract_time_and_feature

def train_and_save_model(n_runs=5, n_classes=10, seed_offset=0, models_dir='models'):
    print(f"Training {n_runs} models with {n_classes} classes...")

    os.makedirs(models_dir, exist_ok=True)
    # Find existing best model for this class count
    existing_best_acc = 0.0
    existing_best_file = None
    
    pattern = re.compile(f"widedeep_K{n_classes}_run\\d+_acc(\\d+)_(\\d+)\\.pt")
    
    # Check for existing models
    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            acc_parts = match.groups()
            acc = float(f"{acc_parts[0]}.{acc_parts[1]}")
            if acc > existing_best_acc:
                existing_best_acc = acc
                existing_best_file = os.path.join(models_dir, filename)
    
    if existing_best_file:
        print(f"Found existing best model: {existing_best_file} (Accuracy: {existing_best_acc:.4f})")
    else:
        print(f"No existing models found for {n_classes} classes")
    
    
    df = pd.read_csv('train_data_reconstructed.csv')
    df['log_t0_o'] = np.log1p(df['t_0_pctg_outage'])
    original_log_t0_o = df['log_t0_o'].copy()
    original_outage_class = df['outage_class'].copy() if 'outage_class' in df.columns else None
    
    results = []
    best_run_acc = 0.0
    any_model_saved = False
    
    for run in range(n_runs):
        print(f"\n{'='*50}")
        print(f"STARTING RUN {run+1}/{n_runs} WITH {n_classes} CLASSES")
        print(f"{'='*50}")
        seed = 42 + seed_offset + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        quantile_bounds = np.quantile(df['log_t0_o'], np.linspace(0, 1, n_classes + 1))
        print(f"Quantile thresholds for {n_classes} classes:", quantile_bounds)
        
        df['outage_class'] = pd.cut(
            df['log_t0_o'],
            bins=quantile_bounds,
            labels=list(range(n_classes)),
            include_lowest=True
        ).astype(int)
        
        deep_cols, wide_cols = get_routed_columns(df.columns, FEATURE_GROUPS)
        feature_order = sorted(set([extract_time_and_feature(c)[1] for c in deep_cols if isinstance(extract_time_and_feature(c), tuple)]))
        deep_features = len(feature_order)
        
        model = WideAndDeepClassifier(
            time_steps=10, 
            deep_features=deep_features,
            wide_features=len(wide_cols),
            hidden_size=64, 
            lstm_units=16, 
            num_classes=n_classes
        )
        
        train_data, val_data = split(df, deep_cols, wide_cols, label_col='outage_class')
        best_model_state, best_val_acc, preds_df = train_model_with_saving(
            model, train_data, val_data, 
            df=df,
            deep_cols=deep_cols,
            wide_cols=wide_cols,
            num_epochs=100, 
            batch_size=32,
            patience=10
        )
        
        run_name = f"run{run+1}_acc{best_val_acc:.4f}".replace(".", "_")
        preds_csv = os.path.join(models_dir, f"all_predictions_K{n_classes}_{run_name}.csv")
        preds_df.to_csv(preds_csv, index=False)
        print(f"Saved all predictions (train + validation) to {preds_csv}")

        if best_val_acc > best_run_acc:
            best_run_acc = best_val_acc
            
        acc_str = f"{best_val_acc:.4f}".replace(".", "_")
        model_filename = os.path.join(models_dir, f"widedeep_K{n_classes}_run{run+1}_acc{acc_str}.pt")
        
        saved_this_model = False
        if not existing_best_file and run == 0:
            torch.save(best_model_state, model_filename)
            print(f"First model saved to {model_filename}")
            existing_best_acc = best_val_acc
            existing_best_file = model_filename
            saved_this_model = True
            any_model_saved = True
        # Better than existing best, save it
        elif best_val_acc > existing_best_acc:
            torch.save(best_model_state, model_filename)
            print(f"New best model! Saved to {model_filename}")
            existing_best_acc = best_val_acc
            existing_best_file = model_filename
            saved_this_model = True
            any_model_saved = True
        else:
            print(f"Model achieved {best_val_acc:.4f} accuracy, but didn't exceed existing best ({existing_best_acc:.4f})")
            model_filename = f"[NOT_SAVED]_widedeep_K{n_classes}_run{run+1}_acc{acc_str}.pt"
        
        results.append({
            'run': run + 1,
            'classes': n_classes,
            'accuracy': best_val_acc,
            'model_file': model_filename,
            'saved': saved_this_model
        })
    
    df['log_t0_o'] = original_log_t0_o
    if original_outage_class is not None:
        df['outage_class'] = original_outage_class
    
    print("\n==== TRAINING SUMMARY ====")
    for r in results:
        save_status = "SAVED" if r['saved'] else "NOT SAVED"
        print(f"Run {r['run']}: Accuracy = {r['accuracy']:.4f}, {save_status}")

    if not any_model_saved and len(results) > 0:
        print("\nNote: No models were saved as none exceeded the existing best model's accuracy.")
    
    if existing_best_file:
        print(f"\nBest model overall: {existing_best_file} (Accuracy: {existing_best_acc:.4f})")
        return existing_best_file
    else:
        return "No model saved"


def create_data_loader(data, batch_size=32, shuffle=True):
    x_deep, x_wide, y = data
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_deep), 
        torch.tensor(x_wide), 
        torch.tensor(y)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model_with_saving(model, train_data, val_data, df, deep_cols, wide_cols, 
                           num_epochs=100, batch_size=32, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_loader = create_data_loader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_data, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    counter = 0
    best_preds = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for i, (deep_x, wide_x, y) in enumerate(train_loader):
            deep_x, wide_x, y = deep_x.to(device), wide_x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(deep_x, wide_x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_preds = []
        all_true = []
        all_indices = []
        
        with torch.no_grad():
            batch_idx = 0
            for deep_x, wide_x, y in val_loader:
                deep_x, wide_x, y = deep_x.to(device), wide_x.to(device), y.to(device)
                outputs = model(deep_x, wide_x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y.cpu().numpy())
                start_idx = batch_idx * batch_size
                end_idx = start_idx + y.size(0)
                all_indices.extend(list(range(start_idx, end_idx)))
                batch_idx += 1
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch [{epoch}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_preds = {
                'indices': all_indices,
                'true_class': all_true,
                'pred_class': all_preds
            }
            counter = 0
            print(f"New best model! Val Acc: {best_val_acc:.4f}")
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    full_data_predictions = predict_all_data(model, train_data, val_data, df, device, batch_size)
    
    return best_model_state, best_val_acc, full_data_predictions


def predict_all_data(model, train_data, val_data, df, device, batch_size=32):
    model.eval()
    
    # Combine train and validation data for prediction
    x_deep_train, x_wide_train, y_train = train_data
    x_deep_val, x_wide_val, y_val = val_data
    x_deep_all = np.vstack([x_deep_train, x_deep_val])
    x_wide_all = np.vstack([x_wide_train, x_wide_val])
    y_all = np.concatenate([y_train, y_val])
    
    all_dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_deep_all), 
        torch.tensor(x_wide_all), 
        torch.tensor(y_all)
    )
    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    all_preds = []
    
    with torch.no_grad():
        for deep_x, wide_x, _ in all_loader:
            deep_x, wide_x = deep_x.to(device), wide_x.to(device)
            outputs = model(deep_x, wide_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    pred_df = pd.DataFrame({
        'true_class': y_all,
        'pred_class': all_preds,
    })
    
    if 'county' in df.columns:
        pred_df['county'] = df['county'].values
    if 'date' in df.columns:
        pred_df['date'] = df['date'].values
    elif 'timestamp' in df.columns:
        pred_df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
    else:
        pred_df['date'] = "2021-02-15"
    
    train_size = len(y_train)
    pred_df['dataset'] = ['train'] * train_size + ['validation'] * len(y_val)
    
    return pred_df


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    best_model_file = train_and_save_model(n_runs=7, n_classes=10)
    print(f"Best model saved as {best_model_file}")