# Harvest Helper - Project Documentation & Handover Guide

**Version:** 1.0  
**Last Updated:** September 15, 2025

## Table of Contents

1. [**Introduction**](#1-introduction)
    * [Project Vision](#11-project-vision)
    * [Core Functionality](#12-core-functionality)
2. [**Features Overview**](#2-features-overview)
    * [Crop Recommendation](#21-crop-recommendation)
    * [Yield Prediction](#22-yield-prediction)
    * [Fertilizer Recommendation](#23-fertilizer-recommendation)
    * [Prediction History](#24-prediction-history)
    * [Model Details & Training](#25-model-details--training)
3. [**Technical Architecture & Stack**](#3-technical-architecture--stack)
    * [Technology Stack](#31-technology-stack)
    * [System Architecture](#32-system-architecture)
    * [Project Structure](#33-project-structure)
4. [**Setup and Deployment**](#4-setup-and-deployment)
    * [Prerequisites](#41-prerequisites)
    * [Local Development Setup](#42-local-development-setup)
    * [Running the Application](#43-running-the-application)
5. [**Codebase Deep Dive**](#5-codebase-deep-dive)
    * [Backend (`app.py`)](#51-backend-apppy)
    * [Training Pipelines](#52-training-pipelines)
    * [Prediction Pipelines](#53-prediction-pipelines)
    * [Database (`database.py`)](#54-database-databasepy)
    * [Frontend (`templates/`)](#55-frontend-templates)
6. [**Data Flow and Database Schema**](#6-data-flow-and-database-schema)
    * [Prediction Data Flow (Example)](#61-prediction-data-flow-example)
    * [Database Schema](#62-database-schema)
7. [**Future Enhancements & Maintenance**](#7-future-enhancements--maintenance)
    * [Potential New Features](#71-potential-new-features)
    * [Maintenance Notes](#72-maintenance-notes)

---

## 1. Introduction

### 1.1. Project Vision

Harvest Helper aims to be an intuitive, AI-powered digital assistant for the agricultural sector. It bridges the gap between complex data science and practical farming by providing clear, actionable recommendations on crop selection, yield forecasting, and fertilizer management. The goal is to help users optimize their practices, increase productivity, and promote sustainable farming.

### 1.2. Core Functionality

The application is a web-based platform with three main predictive models and supporting features for data management and model evaluation.

---

## 2. Features Overview

### 2.1. Crop Recommendation

- **Functionality:** Recommends the most suitable crop based on soil and environmental data.
* **Inputs:** Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall.
* **Output:** The name of the recommended crop (e.g., "Rice").
* **Model:** `RandomForestClassifier`.

### 2.2. Yield Prediction

- **Functionality:** Predicts the total production yield (in tonnes) for a given crop and area.
* **Inputs:** Area (hectares), State, District, Season, Crop, Rainfall, Soil Type.
* **Output:** Predicted yield in tonnes and a calculated yield per hectare.
* **Model:** `RandomForestRegressor`.
* **Special Feature:** The District dropdown is dynamically populated based on the selected State, preventing invalid location entries.

### 2.3. Fertilizer Recommendation

- **Functionality:** Suggests the appropriate fertilizer to use.
* **Inputs:** Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous.
* **Output:** The name of the recommended fertilizer (e.g., "Urea").
* **Model:** `RandomForestClassifier` with SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced data.

### 2.4. Prediction History

- **Functionality:** Allows users to view a log of all past predictions.
* **UI:** A clean, tabbed interface separating Crop, Yield, and Fertilizer history.
* **Features:**
  * Displays all inputs and the corresponding prediction for each historical entry.
  * Provides a "Delete" button for each individual log.
  * Provides a "Delete All" button to clear the entire history for a specific category.
* **Backend:** All records are stored in a local SQLite database (`harvest_helper.db`).

### 2.5. Model Details & Training

- **Functionality:** Provides transparency into the performance of the underlying machine learning models.
* **UI:** Displays a card for each of the three models, showing:
  * Key performance metrics (Accuracy, R² Score, MAE, RMSE).
  * The best hyperparameters found during the last training session.
  * The timestamp of the last training run.
  * A collapsible section to view the full classification report.
* **Features:** A **"Train All Models"** button triggers the retraining of all models, including hyperparameter tuning with `GridSearchCV`.

---

## 3. Technical Architecture & Stack

### 3.1. Technology Stack

- **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Imblearn, Pandas, NumPy
* **Frontend:** HTML, Tailwind CSS, JavaScript
* **Database:** SQLite
* **Serialization:** Pickle (for models), JSON (for performance logs)

### 3.2. System Architecture

The application follows a standard client-server model. The user interacts with the frontend, which sends asynchronous requests (via JavaScript `fetch`) to the Flask backend. The backend processes the request, calls the appropriate prediction pipeline, logs the result to the database, and returns a JSON response to the frontend.

```
+---------------+      HTTP Request      +-----------------+      Python Call      +----------------------+
|               | ---------------------> |                 | --------------------> |                      |
|   Frontend    | (User Input via Form)  |    Flask App    | (Prediction Request)  | Prediction Pipeline  |
| (HTML/JS)     |                        |    (app.py)     |                       | (*_prediction_*.py)  |
|               | <--------------------- |                 | <-------------------- |                      |
+---------------+      JSON Response     +-----------------+      Return Value     +----------------------+
                                               |
                                               | Python Call (Logging/Fetching)
                                               v
                                         +---------------+
                                         |               |
                                         |  Database.py  |
                                         |               |
                                         +---------------+
                                               |
                                               | SQL Read/Write
                                               v
                                         +---------------+
                                         | SQLite DB     |
                                         | (.db file)    |
                                         +---------------+
```

### 3.3. Project Structure

The code is organized into logical directories for maintainability.

```
/Harvest-Helper
|
|-- data/                  # Contains the raw CSV datasets.
|-- models/                # Stores trained .pkl models, encoders, and performance logs.
|-- static/                # Contains CSS, images, and other static assets.
|-- templates/             # Contains all user-facing HTML files.
|
|-- app.py                 # The main Flask application file; handles routing and API endpoints.
|-- database.py            # Manages all SQLite database connections and operations.
|
|-- training_pipeline.py       # Script for training the Crop Recommendation model.
|-- yield_training_pipeline.py   # Script for training the Yield Prediction model.
|-- fertilizer_training_pipeline.py # Script for training the Fertilizer Recommendation model.
|
|-- prediction_pipeline.py     # Logic for making Crop Recommendation predictions.
|-- yield_prediction_pipeline.py # Logic for making Yield Prediction predictions.
|-- fertilizer_prediction_pipeline.py # Logic for making Fertilizer Recommendation predictions.
|
|-- augment_data.py        # Utility script to add features to the yield dataset.
|-- requirements.txt       # List of Python dependencies for pip.
|-- README.md / DOCUMENTATION.md # Project documentation.
```

---

## 4. Setup and Deployment

### 4.1. Prerequisites

- Python 3.7 or higher.
* `pip` package manager.

### 4.2. Local Development Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/harvest-helper.git
    cd harvest-helper
    ```

2. **Create and Activate a Virtual Environment:**

    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Place Datasets:**
    Download the required datasets from the links in the `README.md` and place them in the `/data` folder.
5. **Generate Augmented Data:**
    Run the augmentation script for the yield dataset.

    ```bash
    python augment_data.py
    ```

### 4.3. Running the Application

1. **Train Models:** Before the first launch, all models must be trained. Run these scripts in order.

    ```bash
    python training_pipeline.py
    python yield_training_pipeline.py
    python fertilizer_training_pipeline.py
    ```

2. **Launch the Flask Server:**

    ```bash
    python app.py
    ```

3. **Access the App:** Open a web browser and go to `http://127.0.0.1:5000/`.

---

## 5. Codebase Deep Dive

### 5.1. Backend (`app.py`)

- **Routes:** Defines all URL endpoints.
  * `/`: Renders the homepage.
  * `/crop-recommend`, `/yield-predict`, `/fertilizer-recommend`, `/history`, `/model_details`: Render the respective HTML pages.
  * `/get_districts/<state>`: An API endpoint that returns a JSON list of districts for the yield prediction form.
  * `/predict_*`: API endpoints that receive form data, call the appropriate prediction pipeline, log the results, and return a JSON prediction.
  * `/delete_*`: API endpoints for handling deletion requests from the history page.
  * `/train_models`: An API endpoint to trigger all training pipelines.
* **Data Loading:** Loads data from CSVs at startup to populate dropdowns dynamically.

### 5.2. Training Pipelines

- Each `*_training_pipeline.py` script follows a similar pattern:
    1. **Load Data:** Reads the relevant CSV from the `/data` folder.
    2. **Preprocess:** Handles missing values, encodes categorical variables (using `LabelEncoder` or `pd.get_dummies`), and applies SMOTE where necessary.
    3. **Split Data:** Splits the data into training and testing sets.
    4. **Hyperparameter Tuning:** Uses `GridSearchCV` to find the best parameters for the model.
    5. **Evaluate:** Calculates performance metrics on the test set.
    6. **Save Artifacts:**
        * Saves the best-trained model/pipeline as a `.pkl` file in `/models`.
        * Saves any necessary encoders or feature lists as `.pkl` files.
        * Appends the performance results to `models/model_performance.json`.

### 5.3. Prediction Pipelines

- Each `*_prediction_pipeline.py` script is responsible for inference:
    1. **Load Artifacts:** Loads the pre-trained model and any required encoders from the `/models` directory.
    2. **Preprocess Input:** Takes raw user input (as a dictionary or list), applies the same encoding/scaling transformations used during training to ensure consistency.
    3. **Predict:** Feeds the processed data into the model to get a prediction.
    4. **Post-process:** Converts the predicted value back into a human-readable format (e.g., using `inverse_transform` on a label encoder).

### 5.4. Database (`database.py`)

- **`init_db()`:** Creates the `harvest_helper.db` file and the three log tables (`crop_logs`, `yield_logs`, `fertilizer_logs`) if they don't exist.
* **`log_*()` functions:** Insert a new row into the appropriate table with the user's inputs and the model's prediction.
* **`fetch_*()` functions:** Retrieve all records from a table, ordered by timestamp, for display on the history page.
* **`delete_*()` functions:** Handle the deletion of single or all records from a specified table.

### 5.5. Frontend (`templates/`)

- **`layout.html`:** The master template containing the header, footer, and links to CSS/JS. All other pages extend this.
* **`index.html`:** The main landing page with cards linking to the three core features.
* **`crop.html`, `yield.html`, `fertilizer.html`:** Contain the input forms for each prediction tool. They include JavaScript for form submission via `fetch` and for dynamically displaying results and animations.
* **`history.html`:** A tabbed layout to display data from the three log tables, with JavaScript to handle tab switching and deletion requests.
* **`model_details.html`:** Displays data from the performance JSON file and includes JavaScript to trigger the `/train_models` API endpoint.

---

## 6. Data Flow and Database Schema

### 6.1. Prediction Data Flow (Example: Yield Prediction)

1. **User fills out the form** in `yield.html` and clicks "Predict Yield".
2. **JavaScript** in `yield.html` intercepts the form submission, gathers the data, and sends a `POST` request to the `/predict_yield` endpoint in `app.py`.
3. The **`/predict_yield` route** in `app.py` receives the form data.
4. It calls the **`predict_crop_yield()` function** from `yield_prediction_pipeline.py`, passing the data as a dictionary.
5. `predict_crop_yield()` loads the trained model, encodes the input data to match the training format, and makes a prediction.
6. The prediction result is returned to `app.py`.
7. `app.py` then calls the **`log_yield_prediction()` function** from `database.py` to save the inputs and the result to the SQLite database.
8. `app.py` returns the prediction to the frontend as a **JSON response**.
9. The **JavaScript** in `yield.html` receives the JSON and updates the result `div` to display the prediction to the user.

### 6.2. Database Schema

- **Database File:** `harvest_helper.db`
* **Tables:**
  * `crop_logs`: Stores inputs and outputs for the crop recommendation model.
  * `yield_logs`: Stores inputs and outputs for the yield prediction model.
  * `fertilizer_logs`: Stores inputs and outputs for the fertilizer recommendation model.
    *(Each table includes an auto-incrementing `id` primary key and a `timestamp` column.)*

---

## 7. Future Enhancements & Maintenance

### 7.1. Potential New Features

- **User Authentication:** Add user accounts so individuals can see their own private history.
* **Pest Recommendation Module:** Implement the fourth planned feature for pest/disease identification.
* **Geospatial Integration:** Use location services (with user permission) to automatically fill in temperature and rainfall data.
* **Data Visualization:** Add charts and graphs to the history page to visualize prediction trends over time.
* **Automated Retraining:** Implement a scheduled task (e.g., a cron job) to periodically retrain the models on newly logged data.

### 7.2. Maintenance Notes

- **Retraining:** The models are static and will not improve over time unless retrained. Use the "Train All Models" button or run the training pipelines manually after significant updates to the datasets.
* **Database Management:** The SQLite database file (`harvest_helper.db`) will grow over time. For a production environment, consider migrating to a more robust database system like PostgreSQL.
* **Dependencies:** Periodically update the packages in `requirements.txt` to their latest stable versions to ensure security and performance.
