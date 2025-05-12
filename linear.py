import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import time

from main import get_routed_columns, prepare_model_inputs, FEATURE_GROUPS

# Function to evaluate model with different number of classes
def evaluate_with_n_classes(n_classes):
    print(f"\n==== Testing with {n_classes} classes ====")
    
    df = pd.read_csv('train_data_reconstructed.csv')
    df['log_t0_o'] = np.log1p(df['t_0_pctg_outage'])
    quantile_points = np.linspace(0, 1, n_classes + 1)
    quantile_bounds = np.quantile(df['log_t0_o'], quantile_points)
    print(f"Outage quantile thresholds for {n_classes} classes:", quantile_bounds)
    df['outage_class'] = pd.cut(
        df['log_t0_o'],
        bins=quantile_bounds,
        labels=range(n_classes),
        include_lowest=True
    ).astype(int)
    
    deep_cols, wide_cols = get_routed_columns(df.columns, FEATURE_GROUPS)
    x_deep, x_wide = prepare_model_inputs(df, deep_cols, wide_cols)
    y = df["outage_class"].values
    
    # Flatten and combine inputs
    num_samples, time_steps, num_features = x_deep.shape
    x_deep_flat = x_deep.reshape(num_samples, time_steps * num_features)
    x_all = np.hstack([x_deep_flat, x_wide])
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y, test_size=0.2, stratify=y, random_state=42
    )

    # start_time = time.time()
    clf = LogisticRegression(max_iter=2000, class_weight='balanced')
    clf.fit(x_train, y_train)
    # train_time = time.time() - start_time
    y_pred = clf.predict(x_test)
    
    # Simple t_1_only model for comparison
    t1_feature_index = None
    for i, col in enumerate(deep_cols):
        if 't_1_pctg_outage' in col:
            t1_feature_index = i
            break
    
    if t1_feature_index is not None:
        t1_values = df['t_1_pctg_outage'].values.reshape(-1, 1)
        
        t1_train, t1_test = train_test_split(
            t1_values, test_size=0.2, random_state=42, stratify=y
        )
        
        t1_model = LogisticRegression(max_iter=1000)
        t1_model.fit(t1_train, y_train)
        t1_pred = t1_model.predict(t1_test)
        t1_acc = (t1_pred == y_test).mean()
        print(f"t_1_pctg_outage only accuracy: {t1_acc:.4f}")
    else:
        print("Warning: t_1_pctg_outage not found in features, skipping t1-only model")
        t1_acc = 0
    
    print(f"Classification Report ({n_classes} classes):")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)
    
    accuracy = (y_pred == y_test).mean()    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {n_classes} Classes (Acc: {accuracy:.4f})")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{n_classes}_classes.png")
    
    return {
        'classes': n_classes,
        'accuracy': accuracy,
        # 'train_time': train_time,
        't1_only_accuracy': t1_acc,
        'report': report
    }

if __name__ == "__main__":
    results = []
    for n_classes in [3, 5, 7, 10, 15]:
        result = evaluate_with_n_classes(n_classes)
        results.append(result)
    
    print("\n==== Summary ====")
    result_df = pd.DataFrame([
        {
            'Classes': r['classes'],
            'Accuracy': r['accuracy'],
            # 'Training Time (s)': r['train_time'],
            't1_only Accuracy': r['t1_only_accuracy'],
            'Acc. Difference': r['accuracy'] - r['t1_only_accuracy']
        }
        for r in results
    ])
    print(result_df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.plot(result_df['Classes'], result_df['Accuracy'], 'o-', label='Full Model')
    plt.plot(result_df['Classes'], result_df['t1_only Accuracy'], 'o--', label='t1_only Model')
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs Number of Classes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_vs_classes.png')
    plt.show()