import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

class TwoLayerXORNetwork:
    def __init__(self, beta=1.0, learning_rate=0.5, random_seed=42):
        self.beta = beta
        self.lr = learning_rate
        
        np.random.seed(random_seed)
        self.W_hidden = np.random.uniform(-1.0, 1.0, (2, 3))
        self.W_output = np.random.uniform(-1.0, 1.0, (1, 3))
        
    def _sigmoid(self, s):
        return 1.0 / (1.0 + np.exp(-self.beta * s))
    
    def _sigmoid_derivative_from_output(self, out):
        return self.beta * out * (1.0 - out)

    def train(self, X, D, epochs=5000):
        N = X.shape[0]
        history = []

        for epoch in range(epochs):
            total_error = 0.0
            
            for n in range(N):
                x_n = X[n]
                d_n = D[n]
                
                # --- ETAP I: Przetwarzanie wejść (Forward Propagation) ---
                x_extended = np.array([1.0, x_n[0], x_n[1]]) # [x^(0), x^(1), x^(2)]
                
                s_hidden = np.dot(self.W_hidden, x_extended) # kształt: (2,)
                v_hidden = self._sigmoid(s_hidden)           # kształt: (2,)
                
                v_extended = np.array([1.0, v_hidden[0], v_hidden[1]]) # [v^(0), v^(1), v^(2)]
                s_output = np.dot(self.W_output, v_extended)          # kształt: (1,)
                y = self._sigmoid(s_output)[0]                        # skalar
                
                error = 0.5 * (d_n - y) ** 2
                total_error += error
                
                # --- ETAP II: Propagacja wsteczna (Backpropagation) ---
                delta_y = -(d_n - y) * self._sigmoid_derivative_from_output(y)
                grad_output = delta_y * v_extended
                
                w3_weights = self.W_output[0, 1:] # w_3^(1), w_3^(2)
                delta_hidden = (delta_y * w3_weights) * self._sigmoid_derivative_from_output(v_hidden)
                grad_hidden = np.outer(delta_hidden, x_extended)
                
                # --- Aktualizacja wag ---
                self.W_output -= self.lr * grad_output.reshape(self.W_output.shape)
                self.W_hidden -= self.lr * grad_hidden
                
            if (epoch + 1) % 100 == 0 or epoch == 0:
                history.append({'epoch': epoch + 1, 'total_error': total_error})
                
        return pd.DataFrame(history)

    def predict(self, X):
        predictions = []
        for x_n in X:
            x_extended = np.array([1.0, x_n[0], x_n[1]])
            s_hidden = np.dot(self.W_hidden, x_extended)
            v_hidden = self._sigmoid(s_hidden)
            v_extended = np.array([1.0, v_hidden[0], v_hidden[1]])
            s_output = np.dot(self.W_output, v_extended)
            y = self._sigmoid(s_output)[0]
            predictions.append(y)
        return np.array(predictions)


if __name__ == "__main__":
    # Hiperparametry
    beta_param = 1.0
    learning_rate_param = 1.0
    epochs_count = 1000
    random_seed_param = 12345

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, "data")
    os.makedirs(base_data_dir, exist_ok=True)
    
    existing_items = os.listdir(base_data_dir)
    max_index = 0
    for item in existing_items:
        if item.startswith("run_"):
            parts = item.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                idx = int(parts[1])
                if idx > max_index:
                    max_index = idx
                    
    next_index = max_index + 1
    run_dir = os.path.join(base_data_dir, f"run_{next_index}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Wyniki tego uruchomienia zostaną zapisane w katalogu:\n-> {run_dir}\n")
    
    hyperparams_filename = os.path.join(run_dir, "hyperparameters.json")
    hyperparameters_dict = {
        "run_id": next_index,
        "learning_rate": learning_rate_param,
        "beta": beta_param,
        "epochs": epochs_count,
        "random_seed": random_seed_param
    }
    with open(hyperparams_filename, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters_dict, f, indent=4)

    print("\n--- Wczytywanie zbiorów danych z pliku konfiguracyjnego ---")
    
    config_file = os.path.join(script_dir, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Plik konfiguracyjny nie znaleziony: {config_file}\n"
                                "Utwórz plik config.json z kluczami 'train_data' i 'test_data'")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    train_df = pd.DataFrame(config_data['train_data'])
    test_df = pd.DataFrame(config_data['test_data'])
    
    train_file = os.path.join(run_dir, "train_data.csv")
    test_file = os.path.join(run_dir, "test_data.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"--- Zbiór uczący ---")
    print(train_df.to_string(index=False))
    
    X_train = train_df[['x1', 'x2']].values
    D_train = train_df['d'].values
    
    print("\nRozpoczęcie procesu uczenia sieci...")
    nn = TwoLayerXORNetwork(beta=beta_param, learning_rate=learning_rate_param, random_seed=random_seed_param)
    history_df = nn.train(X_train, D_train, epochs=epochs_count)
    
    print(f"\n--- Zbiór testowy ---")
    
    X_test = test_df[['x1', 'x2']].values
    
    raw_predictions = nn.predict(X_test)
    test_df['y_pred_raw'] = raw_predictions
    test_df['y_pred_binary'] = (raw_predictions >= 0.5).astype(int)
    
    print("Wyniki działania na zbiorze testującym:")
    print(test_df.to_string(index=False))
    
    print("\n--- Zapisywanie wyników symulacji i generowanie wykresu ---")
    
    history_filename = os.path.join(run_dir, "simulation_history.csv")
    history_df.to_csv(history_filename, index=False)
    
    test_results_filename = os.path.join(run_dir, "test_results.csv")
    test_df.to_csv(test_results_filename, index=False)
    
    weights_filename = os.path.join(run_dir, "final_weights.json")
    simulation_summary = {
        "run_id": next_index,
        "total_epochs": epochs_count,
        "final_error": float(history_df['total_error'].iloc[-1]),
        "W_hidden": nn.W_hidden.tolist(),
        "W_output": nn.W_output.tolist()
    }
    with open(weights_filename, 'w', encoding='utf-8') as f:
        json.dump(simulation_summary, f, indent=4)
    
    plot_filename = os.path.join(run_dir, "learning_curve.png")
    plt.figure(figsize=(8, 5))
    plt.plot(history_df['epoch'], history_df['total_error'], color='b', linewidth=2, label='Całkowity błąd (Error)')
    plt.title(f"Krzywa uczenia (Uruchomienie nr {next_index})\n$\\gamma={learning_rate_param}, \\beta={beta_param}$")
    plt.xlabel("Epoka (Iteracja)")
    plt.ylabel("Wartość błędu kwadratowego")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300)
    plt.close()
