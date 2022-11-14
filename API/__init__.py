from fastapi import FastAPI
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load model
model = joblib.load('../Statics/Spam Detector.dat.gz')


def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(input):
    prediction = predict([input], model)

    if prediction == 'ham':
        status = False
        label = "Not Spam"
    else:
        status = True
        label = "Spam"
    return {
        'label': label,
        'spam': status
    }