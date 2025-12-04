import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

#reading excel file
df = pd.read_excel(r"C:\Users\joelk\Downloads\PlayTennis_Dataset.xlsx")

#separating and assigning feature variable as x and target variable y
X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = df['PlayTennis']

#use ordinal encoder for encoding
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

#building model using categorical naive bayes classifier
nb_model = CategoricalNB()
nb_model.fit(X_encoded, y)

#testing model
test_instance = pd.DataFrame([
    {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}])

# Encoding the test example
test_instance_encoded = encoder.transform(test_instance)

#making predictions
prediction = nb_model.predict(test_instance_encoded)[0]

#Getting prediction probabilities
probabilities = nb_model.predict_proba(test_instance_encoded)[0]
prob_no = probabilities[0]
prob_yes = probabilities[1]

#printing the results
print (f"Test Instance: {test_instance.iloc[0].to_dict()}")
print("Predicted Class:",prediction)
print("Probability (No):" ,prob_no)
print("Probability (Yes):",prob_yes)