"""
================================================================================
ROSSMANN STORE SALES - FLASK WEB APPLICATION
================================================================================
FITUR:
  1. Prediksi penjualan dari input user (Store, Promo, Date, dll)
  2. Perbandingan performa semua model
  3. Informasi dataset dan model
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# ===========================================================================
# PATH SETUP
# ===========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
# ===========================================================================
# FLASK APP
# ===========================================================================
app = Flask(__name__)

# ===========================================================================
# LOAD MODELS & DATA
# ===========================================================================
print("Loading models...")

# Load Preprocessing Assets (Scaler & Columns)
with open(os.path.join(MODEL_DIR, 'app_assets.pkl'), 'rb') as f:
    prep_data = pickle.load(f)

scaler          = prep_data['scaler']
feature_columns = prep_data['feature_columns']

# Store data (untuk lookup informasi toko)
store_df = pd.read_csv(os.path.join(DATA_DIR, 'store.csv'))

# Encoding maps
store_type_map    = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
assortment_map    = {'a': 0, 'b': 1, 'c': 2}
state_holiday_map = {'0': 0, 'a': 1, 'b': 2, 'c': 3}

# Load trained models
lr_model = joblib.load(os.path.join(MODEL_DIR, 'linear_regression.pkl'))

import tensorflow as tf
# Load ANN, Backpropagation, and LSTM models using TensorFlow Keras
ann_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'ann_model.keras'), compile=False
)
bp_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'backprop_model.keras'), compile=False
)
lstm_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'lstm_model.keras'), compile=False
)
scaler_lstm = joblib.load(os.path.join(MODEL_DIR, 'scaler_lstm.pkl'))

# Model comparison results
with open(os.path.join(MODEL_DIR, 'model_results.pkl'), 'rb') as f:
    model_results = pickle.load(f)
comparison_df = model_results['comparison']

# LSTM time series data (untuk prediksi LSTM di web)
with open(os.path.join(MODEL_DIR, 'timeseries_data.pkl'), 'rb') as f:
    ts_data = pickle.load(f)
daily_sales = ts_data['daily_sales']

print("SUCCESS: Semua model berhasil dimuat!")

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def prepare_input(store_id, date_str, promo, open_status,
                  state_holiday, school_holiday):
    """Siapkan input features untuk prediksi — lookup store info dari store.csv"""
    try:
        date = pd.to_datetime(date_str)
        day_of_week = date.dayofweek + 1 # Rossmann format: Mon=1, Sun=7

        # Lookup store info
        store_info = store_df[store_df['Store'] == int(store_id)]
        if store_info.empty:
            return None, f"Store ID {store_id} tidak ditemukan"
        store_info = store_info.iloc[0]

        # Build feature dict sesuai urutan feature_columns
        features = {
            'Store': int(store_id),
            'DayOfWeek': int(day_of_week),
            'Open': int(open_status),
            'Promo': int(promo),
            'StateHoliday': state_holiday_map.get(str(state_holiday), 0),
            'SchoolHoliday': int(school_holiday),
            'StoreType': store_type_map.get(str(store_info['StoreType']), 0),
            'Assortment': assortment_map.get(str(store_info['Assortment']), 0),
            'CompetitionDistance': float(store_info['CompetitionDistance'])
                if pd.notna(store_info['CompetitionDistance']) else 5000.0,
            'CompetitionOpenSinceMonth': float(store_info['CompetitionOpenSinceMonth'])
                if pd.notna(store_info['CompetitionOpenSinceMonth']) else 0,
            'CompetitionOpenSinceYear': float(store_info['CompetitionOpenSinceYear'])
                if pd.notna(store_info['CompetitionOpenSinceYear']) else 0,
            'Promo2': int(store_info['Promo2']),
            'Promo2SinceWeek': float(store_info['Promo2SinceWeek'])
                if pd.notna(store_info['Promo2SinceWeek']) else 0,
            'Promo2SinceYear': float(store_info['Promo2SinceYear'])
                if pd.notna(store_info['Promo2SinceYear']) else 0,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'WeekOfYear': int(date.isocalendar()[1]),
            'Quarter': date.quarter,
        }

        # Buat DataFrame sesuai urutan feature_columns
        df = pd.DataFrame([features])
        df = df[feature_columns]  # pastikan urutan kolom benar

        X_scaled = scaler.transform(df)
        return X_scaled, features

    except Exception as e:
        return None, str(e)


def predict_sales(X_scaled):
    """Prediksi menggunakan semua regression models"""
    try:
        # 1. Linear Regression
        lr_pred = float(lr_model.predict(X_scaled)[0])
        
        # 2. ANN
        ann_pred = float(ann_model.predict(X_scaled, verbose=0)[0][0])
        
        # 3. Backpropagation
        bp_pred = float(bp_model.predict(X_scaled, verbose=0)[0][0])

        # 4. LSTM: prediksi berdasarkan 30 hari terakhir dari data historis
        lookback = model_results.get('lookback', 30)
        last_sales = daily_sales.values[-lookback:]
        last_norm  = scaler_lstm.transform(last_sales.reshape(-1, 1)).flatten()
        lstm_input = last_norm.reshape(1, lookback, 1)
        lstm_pred_norm = lstm_model.predict(lstm_input, verbose=0)[0][0]
        lstm_pred = float(scaler_lstm.inverse_transform([[lstm_pred_norm]])[0][0])

        preds = {
            'linear_regression': max(0, round(lr_pred, 2)),
            'ann': max(0, round(ann_pred, 2)),
            'lstm': max(0, round(lstm_pred, 2)),
            'backpropagation': max(0, round(bp_pred, 2)),
        }
        preds['ensemble'] = round(np.mean(list(preds.values())), 2)
        return preds

    except Exception as e:
        return {'error': str(e)}


# ===========================================================================
# ROUTES
# ===========================================================================

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        store_id       = request.form.get('store', 1)
        date_str       = request.form.get('date', datetime.now().strftime('%Y-%m-%d'))
        promo          = request.form.get('promo', 0)
        open_status    = request.form.get('open', 1)
        state_holiday  = request.form.get('state_holiday', '0')
        school_holiday = request.form.get('school_holiday', 0)

        X_scaled, info = prepare_input(
            store_id, date_str, promo,
            open_status, state_holiday, school_holiday
        )

        if X_scaled is None:
            return render_template('predict.html', error=info)

        predictions = predict_sales(X_scaled)
        if 'error' in predictions:
            return render_template('predict.html', error=predictions['error'])

        return render_template('predict.html',
                               predictions=predictions, input_data=info)

    except Exception as e:
        return render_template('predict.html', error=str(e))


@app.route('/comparison')
def comparison():
    comp = {
        'models': comparison_df['Model'].tolist(),
        'mae':    [round(float(x), 2) for x in comparison_df['MAE']],
        'rmse':   [round(float(x), 2) for x in comparison_df['RMSE']],
        'r2':     [round(float(x), 4) for x in comparison_df['R2']],
    }
    kmeans_info = {
        'silhouette': round(float(model_results['kmeans_silhouette']), 4),
        'k': int(model_results['kmeans_k']),
        'inertia': round(float(model_results['kmeans_inertia']), 0),
    }
    return render_template('comparison.html',
                           comparison=comp, kmeans=kmeans_info)


@app.route('/info')
def info():
    info_data = {
        'dataset': {
            'name': 'Rossmann Store Sales',
            'source': 'https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales',
            'description': 'Prediksi penjualan harian toko retail Rossmann'
        },
        'features': feature_columns,
        'models': {
            'Linear Regression': 'Model baseline — hubungan linier antara features dan sales',
            'ANN': 'Artificial Neural Network — 3 hidden layers (128-64-32 neurons)',
            'LSTM': 'Long Short-Term Memory — time series prediction (lookback 30 hari)',
            'K-Means Clustering': 'Unsupervised clustering — segmentasi toko berdasarkan fitur',
            'Backpropagation': 'Custom NN — explicit gradient computation via tf.GradientTape'
        },
        'metrics': {
            'MAE': 'Mean Absolute Error — rata-rata error absolut',
            'RMSE': 'Root Mean Squared Error — penalti lebih besar untuk error besar',
            'R²': 'R-squared — proporsi variance yang dijelaskan oleh model'
        }
    }
    return render_template('info.html', info=info_data)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        X_scaled, info = prepare_input(
            data.get('store', 1),
            data.get('date', datetime.now().strftime('%Y-%m-%d')),
            data.get('promo', 0), data.get('open', 1),
            data.get('state_holiday', '0'), data.get('school_holiday', 0)
        )
        if X_scaled is None:
            return jsonify({'error': info}), 400
        predictions = predict_sales(X_scaled)
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  ROSSMANN STORE SALES — FLASK WEB APP")
    print("  Visit http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
