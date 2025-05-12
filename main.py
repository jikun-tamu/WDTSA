import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import copy
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

FEATURE_GROUPS = {
    'temperature': 'deep',
    'dew_point': 'deep',
    'humidity': 'deep',
    'wind_speed': 'deep',
    'wind_gust': 'wide',
    'pressure': 'wide',
    'pctg_outage': 'deep'
}

def get_routed_columns(df_columns, feature_routing):
    deep_cols = []
    wide_cols = []

    for col in df_columns:
        if col in ["t_0_pctg_outage", "county", "date", "outage_class"]:
            continue

        match = re.match(r"t_(\d+|0)_(.+)", col)
        if match:
            time_lag, base_feature = match.groups()
            is_t0 = (time_lag == "0")

            if base_feature == "pctg_outage":
                route = feature_routing['pctg_outage']
            else:
                route = feature_routing.get(base_feature, 'wide')
            
            if route == 'deep':
                deep_cols.append(col)
            else:
                wide_cols.append(col)

    return deep_cols, wide_cols

def extract_time_and_feature(col_name):
    match = re.match(r"t_(\d+|0)_(.+)", col_name)
    if match:
        t, f = match.groups()
        return int(t), f
    return None, None

def prepare_model_inputs(df, deep_cols, wide_cols):
    wide_input = df[wide_cols].values.astype(np.float32)
    feature_order = sorted(set([extract_time_and_feature(c)[1] for c in deep_cols]))
    time_steps = sorted(set([extract_time_and_feature(c)[0] for c in deep_cols]), reverse=True)
    deep_tensor = np.empty((len(df), len(time_steps), len(feature_order)), dtype=np.float32)
    for i, t in enumerate(time_steps):
        for j, f in enumerate(feature_order):
            col = f"t_{t}_{f}"
            if col in df.columns:
                deep_tensor[:, i, j] = df[col].values
            else:
                deep_tensor[:, i, j] = 0.0
    return deep_tensor, wide_input

class WideAndDeepClassifier(nn.Module):
    def __init__(self, time_steps, deep_features, wide_features, hidden_size=32, lstm_units=16, num_classes=3):
        super(WideAndDeepClassifier, self).__init__()

        # === Deep branch ===
        self.lstm = nn.LSTM(input_size=deep_features, hidden_size=lstm_units, batch_first=True, bidirectional=True)
        self.deep_dense = nn.Linear(2 * lstm_units, hidden_size)
        self.deep_dropout = nn.Dropout(0.25)
        self.deep_batch_norm = nn.BatchNorm1d(hidden_size)

        # === Wide branch ===
        self.wide_dense = nn.Linear(wide_features, hidden_size)
        self.wide_batch_norm = nn.BatchNorm1d(hidden_size)

        # === Fusion output ===
        self.fusion_dense = nn.Linear(2 * hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x_deep, x_wide):
        # Deep path
        lstm_out, _ = self.lstm(x_deep)
        last_out = lstm_out[:, -1, :]  # take final timestep
        x_deep = F.leaky_relu(self.deep_dense(last_out))
        x_deep = self.deep_batch_norm(x_deep)
        x_deep = self.deep_dropout(x_deep)

        # Wide path
        x_wide = F.leaky_relu(self.wide_dense(x_wide))
        x_wide = self.wide_batch_norm(x_wide)

        # Fusion
        x = torch.cat([x_deep, x_wide], dim=1)
        x = F.relu(self.fusion_dense(x))
        return self.output_layer(x)

class PureLSTMClassifier(nn.Module):
    def __init__(self, time_steps, deep_features, hidden_size=32, lstm_units=16, num_classes=3):
        super(PureLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=deep_features, hidden_size=lstm_units, batch_first=True, bidirectional=True)
        self.dense1 = nn.Linear(2 * lstm_units, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.25)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x_deep, x_wide=None):
        lstm_out, _ = self.lstm(x_deep)
        last_out = lstm_out[:, -1, :]  # Take final timestep
        x = F.leaky_relu(self.dense1(last_out))
        x = self.batch_norm(x)
        x = self.dropout(x)
        
        return self.output_layer(x)

class ExpandedPureLSTMClassifier(nn.Module):
    def __init__(self, time_steps, deep_features, hidden_size=64, lstm_units=48, num_classes=3):
        super(ExpandedPureLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=deep_features, 
                           hidden_size=lstm_units, 
                           batch_first=True, 
                           bidirectional=True)
        self.dense1 = nn.Linear(2 * lstm_units, hidden_size * 2) 
        self.batch_norm1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dense2 = nn.Linear(hidden_size * 2, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.25)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x_deep, x_wide=None):
        lstm_out, _ = self.lstm(x_deep)
        last_out = lstm_out[:, -1, :]
        x = F.leaky_relu(self.dense1(last_out))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

def prepare_all_features_as_deep(df, all_cols):
    feature_order = sorted(set([extract_time_and_feature(c)[1] for c in all_cols]))
    time_steps = sorted(set([extract_time_and_feature(c)[0] for c in all_cols]), reverse=True)
    deep_tensor = np.empty((len(df), len(time_steps), len(feature_order)), dtype=np.float32)
    feature_to_idx = {f: i for i, f in enumerate(feature_order)}
    for i, t in enumerate(time_steps):
        for col in all_cols:
            col_t, col_f = extract_time_and_feature(col)
            if col_t == t:
                j = feature_to_idx[col_f]
                deep_tensor[:, i, j] = df[col].values
    return deep_tensor

def split_all_deep(df, all_cols, label_col='outage_class', test_size=0.2, random_state=42):
    labels = df[label_col].values.astype(np.int64)
    df_train, df_val, y_train, y_val = train_test_split(
        df, labels, test_size=test_size, shuffle=True, random_state=random_state, stratify=labels
    )
    x_deep_train = prepare_all_features_as_deep(df_train, all_cols)
    x_deep_val = prepare_all_features_as_deep(df_val, all_cols)
    x_wide_train = np.zeros((len(df_train), 1), dtype=np.float32)
    x_wide_val = np.zeros((len(df_val), 1), dtype=np.float32)
    return (x_deep_train, x_wide_train, y_train), (x_deep_val, x_wide_val, y_val)

def train_model(model, train_data, val_data, num_epochs=20, lr=1e-3, batch_size=64, class_weights=None, device='cpu'):
    x_deep_train, x_wide_train, y_train = train_data
    x_deep_val, x_wide_val, y_val = val_data
    train_dataset = TensorDataset(torch.tensor(x_deep_train), torch.tensor(x_wide_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(x_deep_val), torch.tensor(x_wide_val), torch.tensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = model.to(device)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=False)
        for x_deep, x_wide, y in train_bar:
            x_deep, x_wide, y = x_deep.to(device), x_wide.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x_deep, x_wide)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            train_bar.set_postfix(loss=loss.item())

        train_acc = correct / total
        train_loss /= total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d} [Val]", leave=False)
            for x_deep, x_wide, y in val_bar:
                x_deep, x_wide, y = x_deep.to(device), x_wide.to(device), y.to(device)
                outputs = model(x_deep, x_wide)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:02d}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'train_loss': train_loss
            }
            torch.save(checkpoint, best_model_path)
            print(f"Model improved! Saved checkpoint at epoch {epoch+1} with val_acc: {val_acc:.4f}")

    if os.path.exists(best_model_path):
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_checkpoint['epoch']} with validation accuracy {best_checkpoint['val_acc']:.4f}")


    # model.load_state_dict(torch.load('./checkpoints/0.8724.pth'))

    x_deep_val, x_wide_val, y_val = val_data
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(x_deep_val), torch.tensor(x_wide_val))
        preds = torch.argmax(outputs, dim=1).numpy()
        y_true = np.array(y_val)

    # Confusion matrix
    # cm = confusion_matrix(y_true, preds)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.title("Confusion Matrix")
    # plt.show()

    print(classification_report(y_true, preds, digits=4))
    return model, best_val_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_with_n_classes_architecture_comparison(n_classes):
    quantile_points = np.linspace(0, 1, n_classes + 1)
    quantile_bounds = np.quantile(df['log_t0_o'], quantile_points)
    print(f"Outage quantile thresholds for {n_classes} classes:", quantile_bounds)
    
    df['outage_class'] = pd.cut(
        df['log_t0_o'],
        bins=quantile_bounds,
        labels=range(n_classes),
        include_lowest=True
    ).astype(int)
    
    all_cols = [col for col in df.columns if re.match(r"t_\d+_.+", col) and 
                col not in ["t_0_pctg_outage", "county", "date", "outage_class"]]
    
    # 1. Original Wide-and-Deep Model
    deep_cols, wide_cols = get_routed_columns(df.columns, FEATURE_GROUPS)
    train_data_widedeep, val_data_widedeep = split(df, deep_cols, wide_cols)
    model_widedeep = WideAndDeepClassifier(time_steps=10, deep_features=5, wide_features=20, 
                                          hidden_size=64, lstm_units=16, num_classes=n_classes)
    _, widedeep_acc = train_model(model_widedeep, train_data_widedeep, val_data_widedeep, 
                                 num_epochs=30, batch_size=32)
    
    wd_params = count_parameters(model_widedeep)
    print(f"Wide-and-Deep parameters: {wd_params:,}")
    
    print(f"Wide-and-Deep model accuracy: {widedeep_acc:.4f}")
    
    # 2. Pure LSTM Model
    all_features = sorted(set([extract_time_and_feature(c)[1] for c in all_cols]))
    feature_count = len(all_features)
    train_data_pure_lstm, val_data_pure_lstm = split_all_deep(df, all_cols)
    model_pure_lstm = PureLSTMClassifier(time_steps=10, deep_features=feature_count, 
                                        hidden_size=64, lstm_units=16, num_classes=n_classes)
    _, pure_lstm_acc = train_model(model_pure_lstm, train_data_pure_lstm, val_data_pure_lstm, 
                                  num_epochs=30, batch_size=32)
    
    lstm_params = count_parameters(model_pure_lstm)
    print(f"Pure LSTM parameters: {lstm_params:,}")
    print(f"Pure LSTM model accuracy: {pure_lstm_acc:.4f}")
    
    # 2.5 Expanded Pure LSTM
    model_expanded_lstm = ExpandedPureLSTMClassifier(time_steps=10, deep_features=feature_count, 
                                                hidden_size=64, lstm_units=48, num_classes=n_classes)
    expanded_lstm_params = count_parameters(model_expanded_lstm)
    print(f"Expanded LSTM parameters: {expanded_lstm_params:,}")
    
    _, expanded_lstm_acc = train_model(model_expanded_lstm, train_data_pure_lstm, val_data_pure_lstm, 
                                     num_epochs=30, batch_size=32)
    print(f"Expanded Pure LSTM model accuracy: {expanded_lstm_acc:.4f}")
    
    # 3. Run the original tests too
    deep_cols_outage_only = [col for col in df.columns if col.endswith('pctg_outage') and not col.startswith('t_0_')]
    wide_cols_outage_only = []
    train_data_outage_only, val_data_outage_only = split(df, deep_cols_outage_only, wide_cols_outage_only)
    model_outage_only = WideAndDeepClassifier(
        time_steps=10, 
        deep_features=1, 
        wide_features=0, 
        hidden_size=64, lstm_units=16, num_classes=n_classes
    )
    _, outage_only_acc = train_model(model_outage_only, train_data_outage_only, val_data_outage_only, 
                                    num_epochs=30, batch_size=32)
    
    # Weather-only model
    deep_cols_no_outage = [col for col in deep_cols if not col.endswith('pctg_outage')]
    wide_cols_no_outage = [col for col in wide_cols if not col.endswith('pctg_outage')]
    feature_order_no_outage = sorted(set([extract_time_and_feature(c)[1] for c in deep_cols_no_outage]))
    deep_features_count = len(feature_order_no_outage)
    train_data_no_outage, val_data_no_outage = split(df, deep_cols_no_outage, wide_cols_no_outage)
    model_no_outage = WideAndDeepClassifier(time_steps=10, deep_features=deep_features_count, 
                                           wide_features=len(wide_cols_no_outage), 
                                           hidden_size=64, lstm_units=16, num_classes=n_classes)
    _, no_outage_acc = train_model(model_no_outage, train_data_no_outage, val_data_no_outage, 
                                  num_epochs=30, batch_size=32)
    
    # T1-only model
    deep_cols_t1_only = ['t_1_pctg_outage']
    wide_cols_t1_only = []
    train_data_t1_only, val_data_t1_only = split(df, deep_cols_t1_only, wide_cols_t1_only)
    model_t1_only = WideAndDeepClassifier(
        time_steps=1, 
        deep_features=1, 
        wide_features=0,
        hidden_size=64, lstm_units=16, num_classes=n_classes
    )
    _, t1_only_acc = train_model(model_t1_only, train_data_t1_only, val_data_t1_only, 
                               num_epochs=30, batch_size=32)
    
    return {
        'classes': n_classes,
        'widedeep_accuracy': widedeep_acc,
        'pure_lstm_accuracy': pure_lstm_acc,
        'expanded_lstm_accuracy': expanded_lstm_acc,
        'outage_only_accuracy': outage_only_acc,
        'weather_only_accuracy': no_outage_acc,
        't1_only_accuracy': t1_only_acc,
    }

def evaluate_time_lag_ablation(max_lags=[1, 3, 5, 7, 9]):
    n_classes = 10
    original_log_t0_o = df['log_t0_o'].copy()
    quantile_points = np.linspace(0, 1, n_classes + 1)
    quantile_bounds = np.quantile(df['log_t0_o'], quantile_points)
    print(f"Outage quantile thresholds for {n_classes} classes:", quantile_bounds)
    
    df['outage_class'] = pd.cut(
        df['log_t0_o'],
        bins=quantile_bounds,
        labels=range(n_classes),
        include_lowest=True
    ).astype(int)
    
    results = []
    for max_lag in max_lags:
        print(f"\n{'='*50}")
        print(f"TRAINING WITH MAXIMUM LAG t-{max_lag}")
        print(f"{'='*50}")
        
        filtered_columns = [col for col in df.columns if not re.match(r"t_(\d+)_.+", col) or 
                           int(re.match(r"t_(\d+)_.+", col).group(1)) <= max_lag]
        
        deep_cols, wide_cols = get_routed_columns(filtered_columns, FEATURE_GROUPS)
        train_data, val_data = split(df, deep_cols, wide_cols)
        feature_order = sorted(set([extract_time_and_feature(c)[1] for c in deep_cols]))
        deep_features = len(feature_order)
        
        model = WideAndDeepClassifier(
            time_steps=max_lag, 
            deep_features=deep_features, 
            wide_features=len(wide_cols),
            hidden_size=64, lstm_units=16, num_classes=n_classes
        )
        
        _, val_acc = train_model(
            model, train_data, val_data, num_epochs=30, batch_size=32
        )
        
        results.append({
            'max_lag': max_lag,
            'accuracy': val_acc,
            'deep_cols': len(deep_cols),
            'wide_cols': len(wide_cols)
        })
    
    df['log_t0_o'] = original_log_t0_o
    result_df = pd.DataFrame(results)
    
    print("\n==== TEMPORAL ABLATION STUDY RESULTS ====")
    print(result_df.to_string(index=False))
    
    result_df.to_csv('temporal_ablation_results.csv', index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['max_lag'], result_df['accuracy'], 'o-', linewidth=2)
    plt.xlabel('Maximum Time Lag (t-N)')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance vs. Time Window Size (K=10)')
    plt.grid(True, alpha=0.3)
    plt.xticks(result_df['max_lag'])
    plt.ylim(bottom=max(0.5, result_df['accuracy'].min() - 0.05))
    plt.tight_layout()
    plt.savefig('temporal_ablation_results.png')
    plt.show()
    
    return result_df

def run_architecture_comparison():
    original_log_t0_o = df['log_t0_o'].copy()
    original_outage_class = df['outage_class'].copy()
    
    results = []
    for n_classes in [3, 5, 7, 10, 15]:
        print(f"\n{'='*50}")
        print(f"STARTING ARCHITECTURE COMPARISON WITH {n_classes} CLASSES")
        print(f"{'='*50}")
        result = evaluate_with_n_classes_architecture_comparison(n_classes)
        results.append(result)
        
    df['log_t0_o'] = original_log_t0_o
    df['outage_class'] = original_outage_class
    
    result_df = pd.DataFrame([
        {
            'Classes': r['classes'],
            'Wide-and-Deep': r['widedeep_accuracy'],
            'Pure LSTM': r['pure_lstm_accuracy'],
            'Expanded LSTM': r['expanded_lstm_accuracy'],
            'Outage-Only': r['outage_only_accuracy'],
            'Weather-Only': r['weather_only_accuracy'],
            't1-Only': r['t1_only_accuracy'],
            'WD vs Pure LSTM': r['widedeep_accuracy'] - r['pure_lstm_accuracy'],
            'WD vs Expanded': r['widedeep_accuracy'] - r['expanded_lstm_accuracy'],  # Add this
        }
        for r in results
    ])
    
    print("\n==== ARCHITECTURE COMPARISON SUMMARY ====")
    print(result_df.to_string(index=False))
    
    result_df.to_csv('architecture_comparison_results.csv', index=False)
    
    plt.figure(figsize=(12, 8))
    plt.plot(result_df['Classes'], result_df['Wide-and-Deep'], 'o-', label='Wide-and-Deep', linewidth=2)
    plt.plot(result_df['Classes'], result_df['Pure LSTM'], 's-', label='Pure LSTM', linewidth=2)
    plt.plot(result_df['Classes'], result_df['Expanded LSTM'], 'x-', label='Expanded LSTM', linewidth=2)
    plt.plot(result_df['Classes'], result_df['Outage-Only'], '^-', label='Outage-Only', linewidth=2)
    plt.plot(result_df['Classes'], result_df['t1-Only'], 'd-', label='t1-Only', linewidth=2)
    random_guess = [1/n for n in result_df['Classes']]
    plt.plot(result_df['Classes'], random_guess, '--', color='gray', alpha=0.6, label='Random Guess')
    plt.xlabel('Number of Classes')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Architecture Comparison', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('architecture_comparison.png')
    
    plt.figure(figsize=(10, 6))
    plt.bar(result_df['Classes'], result_df['WD vs Pure LSTM'], color='blue', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy Difference')
    plt.title('Wide-and-Deep Advantage over Pure LSTM', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('widedeep_advantage.png')
    
    plt.show()
    
    return result_df

def split(df, deep_cols, wide_cols, label_col='outage_class', test_size=0.2, random_state=42):
    labels = df[label_col].values.astype(np.int64)
    df_train, df_val, y_train, y_val = train_test_split(
        df, labels, test_size=test_size, shuffle=True, random_state=random_state, stratify=labels
    )
    x_deep_train, x_wide_train = prepare_model_inputs(df_train, deep_cols, wide_cols)
    x_deep_val, x_wide_val = prepare_model_inputs(df_val, deep_cols, wide_cols)
    return (x_deep_train, x_wide_train, y_train), (x_deep_val, x_wide_val, y_val)

if __name__ == "__main__":
    df = pd.read_csv('train_data_reconstructed.csv')
    df['log_t0_o'] = np.log1p(df['t_0_pctg_outage'])

    quantile_bounds = np.quantile(df['log_t0_o'], [0, 1/3, 2/3, 1])
    print("Fixed outage quantile thresholds:", quantile_bounds)
    df['outage_class'] = pd.cut(
        df['log_t0_o'],
        bins=quantile_bounds,
        labels=[0, 1, 2],
        include_lowest=True
    ).astype(int)
    
    run_mode = "temporal_ablation"  # Set to "standard", "multiclass", "feature_importance", "temporal_ablation", or "architecture"
    
    if run_mode == "architecture":
        # Run architecture comparison
        results = run_architecture_comparison()
    elif run_mode == "multiclass":
        # Run the multi-class comparison
        results = run_multiclass_comparison()
    elif run_mode == "feature_importance":
        # Run the feature importance analysis
        evaluate_feature_importance()
    elif run_mode == "temporal_ablation":
        # Run temporal ablation study
        ablation_results = evaluate_time_lag_ablation(max_lags=[1, 3, 5, 7, 9])
    else:
        # Run the standard model training
        deep_cols, wide_cols = get_routed_columns(df.columns, FEATURE_GROUPS)
        model = WideAndDeepClassifier(
            time_steps=10, deep_features=5, wide_features=20,
            hidden_size=64, lstm_units=16, num_classes=3
        )
        print(model)
        train_data, val_data = split(df, deep_cols, wide_cols)
        model, best_val_acc = train_model(model, train_data, val_data, num_epochs=100, batch_size=32)
        print(f"Best validation accuracy: {best_val_acc:.4f}")
