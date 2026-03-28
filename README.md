Fraud Detection Project

This is a beginner-level fraud detection project that uses a small dataset to predict fraudulent transactions using **Logistic Regression**. The project includes data analysis, visualization, and a simple ML model to detect fraud.



Dataset

The dataset includes the following columns:

- **Amount**: Transaction amount  
- **Time**: Time of transaction  
- **Transaction_Type**: Type of transaction (encoded for ML)  
- **Fraud**: Whether the transaction is fraudulent (`Yes`/`No`)



Features

1. **Data Analysis**
   - Preview of the dataset
   - Average transaction amount calculation
   - Fraud vs Non-Fraud counts visualized with a bar chart

2. **Machine Learning**
   - Encode categorical data
   - Split data into training and test sets
   - Train Logistic Regression model
   - Evaluate model accuracy



How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>


2. Navigate to the project folder:

cd Project-3-Fraud-Detection


3. Install required Python packages (if not already installed):

pip install pandas numpy matplotlib seaborn scikit-learn


4. Run the script:

python3 fraud_detection.py


Output
Average Transaction Amount:

2000.0

Fraud Counts:

Fraud	           Count
0 (Non-Fraud)	    3
1 (Fraud)	    2


Bar Chart:
Model Accuracy:

Accuracy: 0.8  # Example, will depend on your dataset split
