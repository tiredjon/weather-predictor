# 🌤️ Weather Forecasting Model for Tashkent

This project demonstrates a machine learning approach to forecasting the **next day's average temperature** in Tashkent based on historical weather data. It utilizes real-world data from the Meteostat API and applies a regression model to predict temperature trends.

---

## 🧠 Project Overview

- **Goal**: Predict tomorrow’s average temperature in Tashkent using today’s weather data.
- **Model**: Random Forest Regressor
- **Framework**: Python, Streamlit
- **Data Source**: [Meteostat](https://meteostat.net/)
- **Training Period**: July 2020 — July 2025
- **Final Model Performance**:
  - **MAE**: ~0.57°C
  - **RMSE**: ~0.77°C

---

## 🔧 Features Used

- Average temperature (`tavg`)
- Minimum and maximum temperature (`tmin`, `tmax`)
- Precipitation (`prcp`)
- Wind speed (`wspd`)
- Atmospheric pressure (`pres`)
- Day of the week
- Month of the year

---

## 🖥️ Streamlit App

The app provides an interactive interface to manually input weather conditions and receive a temperature prediction.

### 📸 Screenshot

![screenshot](https://your-screenshot-url-if-any.com)

---

## 🚀 How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/your-username/weather-predictor.git
cd weather-predictor