import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Define the record to score
datapoint = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict churn probability
result = pipeline.predict_proba([datapoint])[0, 1]

# Print the result
print(f'Probability of lead conversion: {result:.3f}')