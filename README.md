# H2O.ai Titanic Survival Prediction Project

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
- [Methodology](#methodology)
  - [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Insights from the Leaderboard](#insights-from-the-leaderboard)
- [Results and Insights](#results-and-insights)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Technologies Used](#technologies-used)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Google Colab Notebook](#google-colab-notebook)
- [Contributors](#contributors)
- [Acknowledgments](#acknowledgments)

## Overview
The **H2O.ai Titanic Survival Prediction Project** is a data science initiative designed to predict the survival of Titanic passengers using the powerful AutoML capabilities of the H2O.ai platform. This project showcases how automated machine learning can streamline model development—from data preprocessing to model selection and evaluation—thus enabling rapid experimentation and effective decision-making.

## Dataset
The project utilizes the renowned Titanic dataset, which is also featured in the Kaggle competition:
- **Kaggle Dataset:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Description:** The dataset includes detailed information on Titanic passengers such as age, sex, ticket class, fare, and more.
- **Target Variable:** `Survived`
- **Key Features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`
- **Size:** Approximately 891 rows and 12 columns (in the training set)

## Project Files
This repository contains the following key files:
- `h2o_ai_report.pdf`: A comprehensive report detailing the project's methodology, experiments, and performance analysis.
- `h2o_ai_presentation.pptx`: Presentation slides summarizing the project's goals, methodology, and insights.
- **Google Colab Notebook:** An interactive notebook that walks through data preprocessing, model training with H2O.ai AutoML, and evaluation.

## Methodology

### Data Exploration and Preprocessing
- **Data Loading:** The Titanic dataset is imported using H2O.ai’s `import_file` function.
- **Preprocessing Steps:**
  - Conversion of the target variable `Survived` into a categorical format.
  - Removal of non-predictive columns such as `Name`, `Ticket`, `Cabin`, and `PassengerId`.
  - Splitting the data into training (60%), validation (20%), and testing (20%) sets.

### Model Training
- **Initialization:** The H2O.ai platform is initialized to establish the computational environment.
- **AutoML Execution:** H2O’s `H2OAutoML` is employed with a runtime limit of 30 seconds, during which it automatically explores multiple algorithms.
- **Algorithm Selection:** AutoML evaluates various models (including Stacked Ensembles, Gradient Boosting, and XGBoost) and ranks them based on performance metrics such as AUC.

### Model Evaluation
- **Leaderboard Generation:** The trained models are ranked using a leaderboard that provides key performance metrics.
- **Performance Metrics:** Evaluation focuses on metrics like AUC and accuracy, allowing for a clear comparison between models.

### Insights from the Leaderboard
- **Best Performing Model:** Stacked Ensemble models often emerge as the top performers, leveraging the strengths of multiple algorithms.
- **Transparent Comparison:** The leaderboard facilitates a straightforward comparison of the models’ performance, highlighting the robustness of the selected approach.

## Results and Insights
- **High-Performance Models:** The project demonstrates that H2O.ai’s AutoML can quickly produce robust models with competitive performance metrics.
- **Efficiency and Automation:** Automated feature engineering, model selection, and ensembling significantly reduce development time.
- **Actionable Insights:** The results underscore the effectiveness of ensemble methods and provide a framework for rapid prototyping in real-world scenarios.

## Limitations and Future Improvements
- **Runtime Constraints:** The 30-second AutoML runtime may restrict model depth; longer training times could allow for more extensive model exploration.
- **Validation Strategy:** Occasional warnings related to validation frame usage indicate potential for refining cross-validation methods.
- **Future Enhancements:**
  - Extend the training duration for deeper model exploration.
  - Incorporate advanced feature engineering and hyperparameter tuning.
  - Explore additional data sources to enrich the model input.

## Technologies Used
- **H2O.ai:** For automated machine learning and model training.
- **Python:** The primary programming language used for scripting and data manipulation.
- **Pandas & NumPy:** For efficient data processing.
- **Jupyter Notebook / Google Colab:** For interactive coding and experimentation.

## How to Run the Project Locally
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/H2O_ai-Titanic_Dataset-AutoML.git
   cd H2O_ai-Titanic_Dataset-AutoML
   ```
2. **Install Required Libraries:**
   Ensure you have Python installed, then run:
   ```bash
   pip install h2o pandas numpy jupyter
   ```
3. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Open the `H2O_ai_Project.ipynb` notebook and run all cells sequentially.


## Google Colab Notebook
You can also execute the project on Google Colab:
[Google Colab Notebook](https://colab.research.google.com/drive/1qT0lIhK2tMSSvxCDYbGXek1K6lGZRQB4?usp=sharing)

## Contributors
- **Mohamed Amine Boudjemaa** – Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana
- **Abdelkarim Douadjia** – Master 1 Artificial Intelligence, Djilali Bounaama University of Khemis Miliana

## Acknowledgments
- **H2O.ai:** For providing a robust AutoML platform that streamlines the model development process.
- **Kaggle:** For hosting the Titanic dataset and fostering a vibrant data science community.
- **University Faculty:** For their guidance, support, and academic resources.
- **Peers and Instructors:** For their valuable feedback and insights throughout the project.

---

This project exemplifies how automated machine learning can effectively bridge the gap between theoretical concepts and practical applications, delivering high-performance models with minimal manual intervention.
