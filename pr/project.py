import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define a class for the Diabetes dataset
class DiabetesDataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def explore_data(self):
        print("First five rows of the dataset:")
        print(self.data.head())
        print("\nSummary statistics:")
        print(self.data.describe())
        print("\nMissing values:")
        print(self.data.isnull().sum())

    def clean_data(self):
        # Assuming we want to drop rows with any missing values
        self.data = self.data.dropna()

    def visualize_data(self):
        # Plot the distribution of the target variable
        sns.histplot(self.data['Diabetes_012'], kde=False)
        plt.title('Distribution of Diabetes Indicator')
        plt.show()

        # Visualize the relationship between BMI and Diabetes
        sns.scatterplot(x='BMI', y='Diabetes_012', data=self.data)
        plt.title('BMI vs. Diabetes Indicator')
        plt.show()

    def build_regression_model(self):
        # Assuming 'Diabetes_012' is our target variable
        X = self.data.drop('Diabetes_012', axis=1)
        y = self.data['Diabetes_012']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate and print the mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error: {mse}')
        return model, X_test, y_test, mse

    def plot_regression_results(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.show()



# Main function to run the project steps
def main():
    file_path = 'diabetes_012_health_indicators_BRFSS2015.csv'
    dataset = DiabetesDataset(file_path)
    dataset.explore_data()
    dataset.clean_data()
    dataset.visualize_data()
    model, X_test, y_test, mse = dataset.build_regression_model()
    print(f"The Mean Squared Error of the model is: {mse}")
    dataset.plot_regression_results(model, X_test, y_test)

# Run the main function
if __name__ == "__main__":
    main()
