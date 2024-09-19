# Salary Prediction using Linear Regression

This project implements a simple linear regression model to predict an employee's salary based on their years of experience. It utilizes the `Salary_Data.csv` dataset and the `scikit-learn` library for training and testing the model.

## Files

- `ML-Salary-Predictor.ipynb`: Python script that trains the linear regression model and plots the results.
- `Salary_Data.csv`: Dataset containing employee experience and corresponding salary data.

## Dataset

The `Salary_Data.csv` file contains two columns:

- **YearsExperience**: Number of years an employee has worked.
- **Salary**: Corresponding salary of the employee.

### Example of the dataset:
| YearsExperience | Salary  |
|-----------------|---------|
| 1.1             | 39343   |
| 1.3             | 46205   |
| 1.5             | 37731   |
| ...             | ...     |

## Requirements

To run the project, you need the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install these libraries using the following command:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## How to Run

1. Ensure the dataset `Salary_Data.csv` is in the same directory as the Python script.
2. Run the Python script:

```bash
python simple_linear_regression.py
```

The script will:
- Load the dataset.
- Split the data into training and testing sets.
- Train a linear regression model.
- Predict salaries for the test set.
- Plot the results.

## Explanation of the Script

1. **Data Loading**: 
   The dataset is loaded using `pandas`:
   ```python
   dataset = pd.read_csv('Salary_Data.csv')
   ```

2. **Data Splitting**:
   The dataset is split into training and testing sets using `train_test_split`:
   ```python
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)
   ```

3. **Model Training**:
   The linear regression model is trained on the training set:
   ```python
   regressor = LinearRegression()
   regressor.fit(x_train, y_train)
   ```

4. **Prediction**:
   The model predicts salaries for the test set:
   ```python
   y_pred = regressor.predict(x_test)
   ```

5. **Visualization**:
   The script plots the test data against the predicted salaries and displays a regression line:
   ```python
   plt.scatter(x_test, y_test, color='red')
   plt.plot(x_train, regressor.predict(x_train))
   plt.show()
   ```

## Output

- A scatter plot showing the actual salaries versus the predicted salaries.
- A line representing the linear regression fit for the data.

## License

This project is open-source and free to use for educational purposes.
```

This `README.md` provides clear instructions on how to run the script, what to expect, and the purpose of the files in the project.
