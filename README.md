Sales Prediction Web Application using Machine Learning

I recently worked on building and deploying a Sales Prediction Model that uses Linear Regression to predict product sales based on advertising budgets across TV, Radio, and Newspaper.

🔹 What I Did:

Data Preprocessing & Analysis

Collected and cleaned the dataset to handle missing values.

Performed Exploratory Data Analysis (EDA) using Matplotlib & Seaborn, including correlation heatmaps and pairplots to understand relationships between features.

Model Development

Implemented a Linear Regression model using Scikit-Learn.

Trained and tested the model on the dataset with train-test split.

Evaluated the model with metrics like Mean Squared Error (MSE) and R² Score.

Model Deployment

Saved the trained model using Pickle for real-time predictions.

Developed a Flask Web Application to allow users to input advertising budgets and instantly view predicted sales.

Integrated input scaling and percentage prediction capped at 100% for realistic results.

Enhanced the UI using Tailwind CSS for a clean and modern user interface.

Added a dynamic sales vs. TV budget graph using Matplotlib, displayed directly on the website.

🔹 Tools & Technologies Used:

Python, Pandas, NumPy, Matplotlib, Seaborn – for data analysis & visualization

Scikit-Learn – for machine learning model

Flask – for web application backend

Tailwind CSS – for frontend UI

Pickle – for model serialization

🔹 Outcome:

✅ Users can now enter advertising budgets for TV, Radio, and Newspaper, and the system instantly predicts expected sales along with a percentage indicator (relative to maximum sales).
✅ Delivered as a fully functional web app, making machine learning predictions more interactive, user-friendly, and practical.
