import numpy as np
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import streamlit as st

# 1. Загрузка данных
start = datetime(2020, 7, 14)
end = datetime(2025, 7, 14)
tashkent = Point(41.2995, 69.2401)

df = Daily(tashkent, start, end).fetch()

# 2. Предобработка
df.drop(['snow', 'wdir', 'wpgt', 'tsun'], axis=1, inplace=True)
df['prcp'] = df['prcp'].fillna(0)
df['target'] = df['tavg'].shift(-1)
df = df.reset_index(drop=False)
df.dropna(inplace=True)

# 3. Добавим временные признаки
df['month'] = df['time'].dt.month
df['dayofweek'] = df['time'].dt.dayofweek

# 4. Выбор признаков и целевой переменной
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'month', 'dayofweek']
target = 'target'

# 5. Сплит по времени
df['time'] = pd.to_datetime(df['time'])
train = df[df['time'] < '2023-07-15']
val   = df[(df['time'] >= '2023-07-15') & (df['time'] < '2024-07-15')]
test  = df[df['time'] >= '2024-07-15']

X_train, y_train = train[features], train[target]
X_val, y_val = val[features], val[target]
X_test, y_test = test[features], test[target]

# 6. Переобучение модели на всём датасете
full_X = df[features]
full_y = df['target']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(full_X, full_y)

# 7. Оценка модели
test_preds = model.predict(X_test)
mae = mean_absolute_error(y_test, test_preds)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))


st.title("🌤️ Прогноз погоды на завтра (Ташкент)")

st.markdown("Введите текущие погодные параметры:")

# Поля ввода
tavg = st.number_input("Средняя температура (°C)", value=25.0)
tmin = st.number_input("Минимальная температура (°C)", value=20.0)
tmax = st.number_input("Максимальная температура (°C)", value=30.0)
prcp = st.number_input("Осадки (мм)", value=0.0)
wspd = st.number_input("Скорость ветра (км/ч)", value=10.0)
pres = st.number_input("Атм. давление (гПа)", value=1010.0)
month = st.slider("Месяц", 1, 12, 7)
dayofweek = st.slider("День недели (0=Пн, 6=Вс)", 0, 6, 2)

# Прогноз
if st.button("Предсказать"):
    features = np.array([[tavg, tmin, tmax, prcp, wspd, pres, month, dayofweek]])
    prediction = model.predict(features)[0]
    st.success(f"🌡️ Прогноз средней температуры на завтра: **{prediction:.2f}°C**")



# 8. Сохранение модели
print("Сохраняем в:", os.getcwd())
joblib.dump(model, 'weather_model_final.pkl')