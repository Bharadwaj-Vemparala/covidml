import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Title of the app
st.title('COVID-19 Cured Cases Prediction using Random Forest')

# Load dataset
csv_url = 'https://raw.githubusercontent.com/Bharadwaj-Vemparala/covidml/main/covid_19_india.csv'  # Update this path
data = pd.read_csv(csv_url)

# Fixed features: 'Confirmed' and 'Deaths'
features = ['Confirmed', 'Deaths']
target = 'Cured'

# Ensure 'Cured' is a valid column
if target not in data.columns:
    st.error("'Cured' column not found in the dataset. Please check the dataset.")
else:
    # Split the data into features (X) and target (y)
    X = data[features]
    y = data[target]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    st.sidebar.subheader('Model Training')

    if 'model' not in st.session_state:
        st.session_state.model = None

    # Train the model only when the button is clicked
    if st.sidebar.button('Train Random Forest Model'):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.write('### Model trained successfully!')

        # Model Evaluation
        y_pred = model.predict(X_test)
        st.write(f"**Mean Absolute Error:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**R-squared Score:** {r2_score(y_test, y_pred):.2f}")

    # Section for user input and prediction
    st.subheader('Make Predictions')

    # Allow user to input values for the fixed features 'Confirmed' and 'Deaths'
    confirmed = st.number_input("Enter value for Confirmed cases", step=1.0, format="%.2f")
    deaths = st.number_input("Enter value for Deaths", step=1.0, format="%.2f")

    # Predict based on user input if model is trained
    if st.button('Predict'):
        if st.session_state.model is not None:
            user_input = [[confirmed, deaths]]
            prediction = st.session_state.model.predict(user_input)
            st.write(f'### Predicted Cured Cases: {prediction[0]:.2f}')
        else:
            st.write('Please train the model before making predictions.')
