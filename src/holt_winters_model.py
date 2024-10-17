import matplotlib
matplotlib.use('Agg')  # Используем 'Agg' для отрисовки графиков без графического интерфейса

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из Excel-файла
file_path = 'data/Urov_11subg_nm.xlsx'
data = pd.read_excel(file_path, sheet_name='Лист1')

# Извлечение данных для "Российская Федерация"
# Поиск строки с названием "Российская Федерация" и извлечение значений по кварталам
russia_data = data.iloc[3, 2:].dropna().reset_index(drop=True)

# Преобразование данных в числовой формат (если есть текстовые данные)
russia_data = pd.to_numeric(russia_data, errors='coerce')

# Создание временного ряда с использованием кварталов
date_range = pd.date_range(start='2014Q1', periods=len(russia_data), freq='QE-DEC')  # Замена Q-DEC на QE-DEC
time_series = pd.Series(russia_data.values, index=date_range)

# Функция для аддитивной модели Хольта-Уинтерса
def holt_winters_additive(series, alpha, beta, gamma, season_length, n_forecast):
    n = len(series)
    level = [series.iloc[0]]  # Инициализация уровня
    trend = [(series.iloc[1] - series.iloc[0])]  # Инициализация тренда
    seasonal = [series.iloc[i] - series.mean() for i in range(season_length)]  # Инициализация сезонности
    forecast = []

    # Цикл для вычисления уровней, тренда и сезонности
    for i in range(1, n):
        if i >= season_length:
            season = seasonal[i % season_length]
        else:
            season = 0

        # Обновление уровня, тренда и сезонного компонента
        new_level = alpha * (series.iloc[i] - season) + (1 - alpha) * (level[-1] + trend[-1])
        new_trend = beta * (new_level - level[-1]) + (1 - beta) * trend[-1]
        new_season = gamma * (series.iloc[i] - new_level) + (1 - gamma) * season

        # Добавление обновленных значений в соответствующие списки
        level.append(new_level)
        trend.append(new_trend)
        if i < season_length:
            seasonal.append(new_season)
        else:
            seasonal[i % season_length] = new_season

    # Прогнозирование на n_forecast шагов вперед
    for i in range(n_forecast):
        m = i + 1
        forecast.append(level[-1] + m * trend[-1] + seasonal[m % season_length])

    return forecast, level, trend, seasonal

# Параметры модели (можно изменять для тюнинга)
alpha = 0.5  # Параметр сглаживания уровня
beta = 0.3   # Параметр сглаживания тренда
gamma = 0.2  # Параметр сглаживания сезонности
season_length = 4  # Длина сезона (4 квартала)
n_forecast = 8  # Прогноз на 8 кварталов вперед

# Применение модели к временным рядам
forecast, level, trend, seasonal = holt_winters_additive(time_series, alpha, beta, gamma, season_length, n_forecast)

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series, label='Actual Data', marker='o')
future_dates = pd.date_range(start=time_series.index[-1] + pd.offsets.QuarterEnd(), periods=n_forecast, freq='QE-DEC')  # Замена Q-DEC на QE-DEC
plt.plot(future_dates, forecast, label='Forecast', marker='o', linestyle='--')
plt.title('Holt-Winters Additive Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('holt_winters_forecast.png')  # Сохранение графика в файл
