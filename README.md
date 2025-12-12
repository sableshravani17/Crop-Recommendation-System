1)Technologies & Libraries Used

Python

Pandas

NumPy

Scikit-learn

RandomForestClassifier

Train-test split

LabelEncoder

Accuracy Score

Matplotlib / Seaborn 

flask

2)Crop-Recommendation-System/
│
├── data/
│   └── crop_recommendation.csv
│
├── model/
│   └── crop_model.pkl
│
├── app.py              # Streamlit/Flask app
├── train_model.py      # ML training script
├── requirements.txt
└── README.md

3)Model Details
Algorithm used: Random Forest Classifier
Trained on soil nutrients (N, P, K), pH, temperature, humidity, rainfall.
Evaluated using accuracy and test split.
Random Forest chosen because it reduces overfitting and performs well on tabular data.

4)How the Project Works
Dataset is loaded and cleaned.
Labels (crop names) are encoded.
Features and target variable are separated.
Data is split into training and testing parts.
Random Forest model is trained.
Model performance is evaluated.
The trained model is saved.
The application takes user input and predicts the crop.
