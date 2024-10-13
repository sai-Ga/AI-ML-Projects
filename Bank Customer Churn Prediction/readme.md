# Bank Customer Churn Prediction

This repository contains a machine learning project focused on predicting customer churn for a bank. The dataset used includes various customer attributes like credit score, age, account balance, and more. The goal of this project is to build a model that can accurately predict whether a customer will leave the bank (churn).

## Dataset

The dataset consists of 10,000 customer records with the following features:

- `customer_id`: Unique identifier for each customer.
- `credit_score`: Customer's credit score.
- `country`: Country of the customer (e.g., France, Spain).
- `gender`: Gender of the customer (Male/Female).
- `age`: Customer's age.
- `tenure`: Number of years the customer has been with the bank.
- `balance`: Customer's account balance.
- `products_number`: Number of products the customer uses at the bank.
- `credit_card`: Whether the customer has a credit card (1: Yes, 0: No).
- `active_member`: Whether the customer is an active member (1: Yes, 0: No).
- `estimated_salary`: Estimated salary of the customer.
- `churn`: Whether the customer has churned (1: Yes, 0: No).

### File Descriptions

- **Bank Customer Churn Prediction.csv**: The main dataset used for training and evaluating the models.
- **Customer Churn Analysis.pbix**: A Power BI report that provides visual insights into customer churn patterns and key performance metrics.

## Project Overview

The project is divided into the following phases:

1. **Exploratory Data Analysis (EDA)**: Understanding the dataset through visualizations and summary statistics.
2. **Feature Engineering**: Creating and selecting features that will improve model performance.
3. **Modeling**: Using machine learning algorithms to predict churn. This includes models like Logistic Regression, Decision Trees, Random Forests, and more.
4. **Model Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, and the confusion matrix.
5. **Deployment**: Steps for deploying the model to a production environment (if applicable).

## Dependencies

To run the project, the following dependencies are required:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Power BI Desktop (for opening `.pbix` files)

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/sai-Ga/AI-ML-Projects.git
   ```

2. Navigate to the project directory and install the necessary packages:
   ```bash
   cd AI-ML-Projects
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook or Python scripts to explore the data and train models.

4. Open the Power BI report (`Customer Churn Analysis.pbix`) to explore the visual analysis.

   ![image](https://github.com/user-attachments/assets/d0a8b90e-9955-40f9-a49f-32c52e6c9166)


## Results

The results of the analysis and model performance metrics will be included in the report. The final model aims to help the bank identify customers who are likely to churn, enabling proactive retention strategies.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the contributors and open-source community.
