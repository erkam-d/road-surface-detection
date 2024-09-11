# importiere benötigte bibliotheken
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

start_time = time.time() # wird benötigt um die laufzeit des skripts zu messen
print("starten des skripts...")

# definiere den pfad zu den daten ÄNDERN NACH BEDARF
base_path = r'C:\Users\easyd\Documents\abgabe_BA\trainingsdata'

# praktische funktion um daten zu laden, labeln und zu 'fenstern'
# fenstern bedeutet, dass die daten in zeitfenster unterteilt werden
# die größe des fensters wird durch window_size definiert

def load_and_window_data(path, label, window_size):
    
    # erste hälfte der funktion das lesen der daten
    # und das zusammenfügen der daten in ein dataframe
    
    print(f"laden und 'fenstern' der {label} daten...")
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(path, f)) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df['Label'] = label
    
    X = df[['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']]
    y = df['Label'].map({'asphalt': 0, 'cobblestone': 1, 'paving_stone': 2})
    
    lstm_data, lstm_labels = [], []
    for i in range(len(X) - window_size + 1):
        lstm_data.append(X.iloc[i:i + window_size].values)
        lstm_labels.append(y.iloc[i + window_size - 1])
    
    return np.array(lstm_data), np.array(lstm_labels)

# definiere fenstergröße
window_size = 30

# laden der daten, labeln und 'fenstern' der daten
X_asphalt, y_asphalt = load_and_window_data(os.path.join(base_path, 'asphalt_processed'), 'asphalt', window_size)
X_cobblestone, y_cobblestone = load_and_window_data(os.path.join(base_path, 'cobblestone_processed'), 'cobblestone', window_size)
X_paving_stone, y_paving_stone = load_and_window_data(os.path.join(base_path, 'paving_stone_processed'), 'paving_stone', window_size)

# kombiniere alle daten in einem array
X_all = np.concatenate([X_asphalt, X_cobblestone, X_paving_stone], axis=0)
y_all = np.concatenate([y_asphalt, y_cobblestone, y_paving_stone], axis=0)

print("alle daten geladen, 'gefenstert' und kombiniert.")

# splitte die fensterdaten in trainings- und testdaten
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_all, y_all, test_size=0.2, random_state=42, shuffle=True)

# definiere das modell
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, X_train_lstm.shape[2]))) # 8 neuronen, window size, 6 features
model.add(Dense(3, activation='softmax'))  # softmax aktivierungsfunktion für multiklassenprobleme

#model.summary() # coole methode um eine übersicht über das modell zu bekommen

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("trainiere das modell...")
history = model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# evaluiere das modell
y_pred_prob = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\nKlassifikation report:")
print(classification_report(y_test_lstm, y_pred))
print("\nKonfusionsmatrix:")
print(confusion_matrix(y_test_lstm, y_pred))

end_time = time.time()

print(f"skript wurde in {end_time - start_time} sekunden ausgeführt.")

import joblib
import tensorflow as tf
print("\nspeichern des modells...")
model.save('modell_name.keras')

# konvertiere das modell zu tflite - experimentelle flags notwendig (flex), unkommentieren falls notwendig
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter._experimental_lower_tensor_list_ops = False # kann kommentiert werden
# converter.experimental_enable_resource_variables = True # kann kommentiert werden
tflite_model = converter.convert()

# Save the TFLite model
with open('model_name.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nmodell wurde konvertiert und erfolgreich gespeichert")