import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

st.title('Сервис для прогноза возврата клиентом кредита')

def load_data():
    data = pd.read_csv('cleaned_df.csv')
    return data

data = load_data()

def train_model():
    X = data.drop('SeriousDlqin2yrs', axis=1)
    y = data['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = DecisionTreeClassifier(max_depth=8, min_samples_split=10, min_samples_leaf=5,max_features='sqrt', random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = roc_auc_score(y_test, y_pred)
    
    return model, accuracy

model, accuracy = train_model()

st.sidebar.header('Данные клиента')

binary_features = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']

integer_features = ['NumberOfDependents', 'age' ]

inputs = {}
for column in data.columns:
    if column != 'SeriousDlqin2yrs':
        if column in binary_features:
            answer = st.sidebar.radio(f'{column}', options=['Нет', 'Да'], index=0)
            inputs[column] = 1 if answer == 'Да' else 0
        elif column in integer_features:
            min_val = int(data[column].min())
            max_val = int(data[column].max())
            default_val = int(data[column].median())
            inputs[column] = st.sidebar.number_input(f'{column}', min_value=min_val, max_value=max_val, value=default_val, step=1)
        elif data[column].dtype == 'float64':
            min_val = float(data[column].min())
            max_val = float(data[column].max())
            default_val = float(data[column].median())
            inputs[column] = st.sidebar.slider(f'{column}', min_val, max_val, default_val)
        elif data[column].dtype == 'int64':
            min_val = int(data[column].min())
            max_val = int(data[column].max())
            default_val = int(data[column].median())
            inputs[column] = st.sidebar.number_input(f'{column}', min_value=min_val, max_value=max_val, value=default_val, step=1)
        elif column == 'GroupAge':
            options = ['a', 'b', 'c', 'd', 'e', 'C']
            selected = st.sidebar.selectbox('GroupAge', options)
            inputs[column] = options.index(selected)

if st.sidebar.button('Рассчитать вероятность возврата кредита'):
    input_df = pd.DataFrame([inputs])

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

st.subheader('О модели')
st.write(f'Точность модели: {accuracy*100:.2f}%')

st.markdown("""
**Описание признаков модели:**

- **RevolvingUtilizationOfUnsecuredLines**: Общий баланс средств
- **age**: Возраст заемщика
- **NumberOfTime30-59DaysPastDueNotWorse**: Имелась ли просрочка в 30-59 дней
- **DebtRatio**: Отношение расходов к месячному доходу в месяц
- **MonthlyIncome**: Доход заемщика в месяц
- **NumberOfOpenCreditLinesAndLoans**: Количество открытых кредитов
- **NumberOfTimes90DaysLate**: Имелась ли просрочка в 90 и более дней
- **RealEstateLoansOrLines**: Количество кредитов под залог недвижимости
- **NumberOfTime60-89DaysPastDueNotWorse**: Имелась ли просрочка в 60-89 дней
- **NumberOfDependents**: Количество иждивенцев
- **GroupAge**: Возрастная группа
""")

st.subheader('Визуализация дерева решений')
fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(model, feature_names=data.drop('SeriousDlqin2yrs', axis=1).columns, class_names=['No Default', 'Default'], filled=True, rounded=True, max_depth=2,ax=ax)
st.pyplot(fig)

st.subheader('Важность признаков')
feature_importances = pd.DataFrame({'Feature': data.drop('SeriousDlqin2yrs', axis=1).columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
st.dataframe(feature_importances)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances['Feature'], feature_importances['Importance'])
ax.set_xlabel('Важность признака')
ax.set_title('Важность признаков в модели')
st.pyplot(fig)

if st.checkbox('Показать исходные данные'):
    st.write(data)





    