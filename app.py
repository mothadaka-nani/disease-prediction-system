
from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd

# Load model and symptom mapping
with open('symptom_disease_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('symptom_mapping.pkl', 'rb') as f:
    symptom_mapping = pickle.load(f)

# Load the CSV files for details
#symptoms_df = pd.read_csv('datasets\symtoms_df.csv')
description_df = pd.read_csv(r'datasets\description.csv')
precautions_df = pd.read_csv(r'datasets\precautions_df.csv')
medications_df = pd.read_csv(r'datasets\medications.csv')
workout_df = pd.read_csv(r'datasets\workout_df.csv')
diets_df = pd.read_csv(r'datasets\diets.csv')


# Function to remove unnecessary characters from list outputs
def clean_list(text):
    return ', '.join([item.strip().replace('[', '').replace(']', '').replace("'", '').replace('"', '') for item in text])

# Function to get details of a disease
def get_disease_details(disease):
    description = description_df[description_df['Disease'] == disease]['Description'].values[0]
    
    precautions_row = precautions_df[precautions_df['Disease'] == disease]
    precautions = clean_list(precautions_row.iloc[:, 2:].values.flatten())
    
    medications_row = medications_df[medications_df['Disease'] == disease]
    medications = clean_list(medications_row['Medication'].values)
    
    workouts_row = workout_df[workout_df['disease'] == disease]
    workouts = clean_list(workouts_row['workout'].values)
    
    diets_row = diets_df[diets_df['Disease'] == disease]
    diets = clean_list(diets_row['Diet'].values)
    
    return {
        "Description": description,
        "Precautions": precautions,
        "Medications": medications,
        "Workouts": workouts,
        "Diets": diets
    }

# Initialize Flask app
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to get diseases based on symptoms
@app.route('/predict', methods=['POST'])
def predict_disease():
    symptoms = request.json.get('symptoms', [])
    
    # Transform the input symptoms
    input_symptoms = symptom_mapping.transform([symptoms])
    
    # Predict the disease
    predicted_diseases = model.predict(input_symptoms)
    
    # Get details for each predicted disease
    result = {}
    for disease in predicted_diseases:
        result[disease] = get_disease_details(disease)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)







# from flask import Flask, request, render_template
# import pandas as pd
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Load datasets
# sym_des = pd.read_csv("datasets/symtoms_df.csv")
# precautions = pd.read_csv("datasets/precautions_df.csv")
# workout = pd.read_csv("datasets/workout_df.csv")
# description = pd.read_csv("datasets/description.csv")
# medications = pd.read_csv('datasets/medications.csv')
# diets = pd.read_csv("datasets/diets.csv")

# # Load model
# svc = pickle.load(open('svc.pkl', 'rb'))

# # Helper functions
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])

#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]

#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]

#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]

#     wrkout = workout[workout['disease'] == dis]['workout']
#     wrkout = [w for w in wrkout.values]

#     return desc, pre, med, die, wrkout

# # Define symptoms dictionary
# symptoms_dict = {
#     'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
#     'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
#     'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
#     'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
#     'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
#     'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
#     'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
#     'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
#     'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
#     'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
#     'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
#     'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
#     'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
#     'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
#     'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 
#     'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
#     'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 
#     'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
#     'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
#     'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
#     'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
#     'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
#     'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
#     'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
#     'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91,
#     'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
#     'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
#     'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
#     'dischromic_patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
#     'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
#     'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
#     'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
#     'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
#     'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
#     'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
#     'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
#     'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
#     'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
# }

# # Define diseases list
# diseases_list = {
#     0: 'Disease A', 1: 'Disease B', 2: 'Disease C', 3: 'Disease D', 
#     4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis',
#     8: 'Chicken pox', 9: 'Chronic cholestasis', 10: 'Common Cold', 
#     11: 'Dengue', 12: 'Diabetes', 13: 'Dimorphic hemorrhoids (piles)', 
#     14: 'Drug Reaction', 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 
#     18: 'Heart attack', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 
#     22: 'Hepatitis E', 23: 'Hypertension', 24: 'Hyperthyroidism', 
#     25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 
#     28: 'Jaundice', 29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthritis', 
#     32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer disease', 
#     34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 
#     37: 'Typhoid', 38: 'Urinary tract infection', 39: 'Varicose veins', 
#     40: 'Vertigo', 41: 'Fungal infection'
#     # Add more diseases as needed
# }


# # Model Prediction function
# def get_top_predicted_values(patient_symptoms, top_n=2):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         if item in symptoms_dict:
#             input_vector[symptoms_dict[item]] = 1
#         else:
#             # Handle unknown symptoms
#             pass

#     # Get probabilities for each class
#     probabilities = svc.decision_function([input_vector])[0]

#     # Get indices of the top_n diseases
#     top_indices = np.argsort(probabilities)[-top_n:][::-1]

#     # Map indices to disease names
#     top_diseases = [diseases_list[i] for i in top_indices]

#     return top_diseases

# # Flask routes
# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route('/predict', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms')
#         print(symptoms)
#         if not symptoms:
#             message = "Please enter at least one symptom."
#             return render_template('index.html', message=message)
#         else:
#             # Split the user's input into a list of symptoms (assuming they are comma-separated)
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             # Remove any extra characters, if any
#             user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]

#             # Get top predicted diseases
#             top_diseases = get_top_predicted_values(user_symptoms, top_n=2)

#             # Prepare data for each predicted disease
#             results = []
#             for predicted_disease in top_diseases:
#                 dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

#                 my_precautions = []
#                 for i in precautions[0]:
#                     my_precautions.append(i)

#                 result = {
#                     'predicted_disease': predicted_disease,
#                     'dis_des': dis_des,
#                     'my_precautions': my_precautions,
#                     'medications': medications,
#                     'my_diet': rec_diet,
#                     'workout': workout
#                 }
#                 results.append(result)

#             return render_template('results.html', results=results, user_symptoms=user_symptoms)

#     return render_template('index.html')

# # Other routes (about, contact, etc.) remain the same

# if __name__ == "__main__":
#     app.run(debug=True)
