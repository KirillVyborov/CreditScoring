import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

rub_usd = 80.0

st.title('Сервис для прогноза возврата клиентом кредита')

def load_data():
    data = pd.read_csv('cleaned_df.csv')
    if 'GroupAge' in data.columns:
        data = data.drop('GroupAge', axis=1)
    return data

data = load_data()

def train_model():
    X = data.drop('SeriousDlqin2yrs', axis=1)
    y = data['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=8, max_features='sqrt', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = roc_auc_score(y_test, y_pred)
    
    return model, accuracy, y_test, y_proba

model, accuracy, y_test, y_proba = train_model()

st.sidebar.header('Данные клиента')

feature_translate = {
    'RevolvingUtilizationOfUnsecuredLines': 'Отношение текущего долга по кредитной карте к общему кредитному лимиту',
    'age': 'Возраст',
    'NumberOfTime30-89DaysPastDueNotWorse': 'Просрочки 30-89 дней',
    'DebtRatio': 'Отношение ежемесячных платежей по долгам к ежемесячному доходу',
    'MonthlyIncome': 'Ежемесячный доход',
    'NumberOfOpenCreditLinesAndLoans': 'Общее количество открытых кредитных линий и активных кредитов',
    'NumberOfTimes90DaysLate': 'Просрочки 90 и более дней',
    'RealEstateLoansOrLines': 'Количество активных ипотечных кредитов',
    'NumberOfDependents': 'Количество иждивенцев'}

max_values = {
    'age': data['age'].max(),
    'MonthlyIncome': data['MonthlyIncome'].max(),
    'RevolvingUtilizationOfUnsecuredLines': data['RevolvingUtilizationOfUnsecuredLines'].max(),
    'NumberOfOpenCreditLinesAndLoans': data['NumberOfOpenCreditLinesAndLoans'].max(),
    'RealEstateLoansOrLines': data['RealEstateLoansOrLines'].max(),
    'NumberOfDependents': data['NumberOfDependents'].max(),
    'NumberOfTime30-89DaysPastDueNotWorse': data['NumberOfTime30-89DaysPastDueNotWorse'].max(),
    'NumberOfTimes90DaysLate': data['NumberOfTimes90DaysLate'].max()}

inputs = {}

inputs['age'] = st.sidebar.number_input('Возраст клиента', min_value=21, value=30, step=1)

debt_payment = st.sidebar.number_input('Плата по долгам (руб/мес)', min_value=0, value=10000)
monthly_income_rub = st.sidebar.number_input('Месячный доход (руб)', min_value=0, value=50000)
inputs['MonthlyIncome'] = monthly_income_rub / rub_usd
inputs['DebtRatio'] = debt_payment / (monthly_income_rub + 0.000001)

credit_limit = st.sidebar.number_input('Кредитный лимит по карте (руб)', min_value=0, value=50000)
current_debt = st.sidebar.number_input('Текущий долг по карте (руб)', min_value=0, value=10000)
inputs['RevolvingUtilizationOfUnsecuredLines'] = current_debt / (credit_limit + 0.000001)

inputs['NumberOfOpenCreditLinesAndLoans'] = st.sidebar.number_input('Количество открытых кредитых линий', min_value=0, value=0, step=1)

inputs['RealEstateLoansOrLines'] = st.sidebar.number_input('Количество ипотечных кредитов', min_value=0, value=0, step=1)

inputs['NumberOfDependents'] = st.sidebar.number_input('Количество иждивенцев', min_value=0, value=0, step=1)

inputs['NumberOfTime30-89DaysPastDueNotWorse'] = 1 if st.sidebar.radio('Были ли просрочки 30-89 дней?', options=['Нет', 'Да'], index=0) == 'Да' else 0
inputs['NumberOfTimes90DaysLate'] = 1 if st.sidebar.radio('Были ли просрочки 90 дней?', options=['Нет', 'Да'], index=0) == 'Да' else 0

if st.sidebar.button('Рассчитать вероятность возврата кредита'):
    input_df = pd.DataFrame([inputs])
    input_df = input_df[data.drop('SeriousDlqin2yrs', axis=1).columns]

    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Результат оценки')
    if prediction[0] == 0:
        st.success('Шанс возврата высокий')
    else:
        st.error('Шанс возврата низкий')
    
    st.write(f'Вероятность возврата кредита: {prediction_proba[0][0]*100:.2f}%')
    
    fig, ax = plt.subplots()
    ax.bar(['Клиент вернет кредит', 'Клиент не вернет кредит'], prediction_proba[0], color=['green', 'red'])
    ax.set_ylabel('Вероятность')
    ax.set_title('Распределение вероятностей')
    st.pyplot(fig)

if st.checkbox('Показать информацию о модели'):
    st.subheader('О модели')
    st.write(f'Точность модели: {accuracy*100:.2f}%')

    st.subheader('Визуализация дерева решений')
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(model, feature_names=[feature_translate.get(col, col) for col in data.drop('SeriousDlqin2yrs', axis=1).columns], 
                  class_names=['No Default', 'Default'], filled=True, rounded=True, max_depth=2, ax=ax)
    st.pyplot(fig)

    st.subheader('Важность признаков')
    feature_importances = pd.DataFrame({
        'Feature': [feature_translate.get(col, col) for col in data.drop('SeriousDlqin2yrs', axis=1).columns],
        'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importances['Feature'], feature_importances['Importance'])
    ax.set_xlabel('Важность признака')
    ax.set_title('Важность признаков')
    st.pyplot(fig)

    st.subheader('Оценка точности модели с помощью ROC-кривой')
    fig, ax = plt.subplots(figsize=(10, 10))
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'Decision Tree (accuracy = {accuracy:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Кривая')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

if st.checkbox('Показать исходные данные'):
    st.write(data)








    