# Import the necessary libraries
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load the data
df = pd.read_csv("./transcriptions.csv")
df.head()


load_dotenv()


# Define the client
client = OpenAI(api_key=os.getenv("OPEN_API_SECRET"))

def extract_info_with_openai(transcription):
    messages=[
        {
            "role": "system", "content": "You are a healthcare professional extracting patient data. Always return both the age and the recommended treatment. If information is missing, create the field and specify 'Unknown'",
            "role": "user", "content": f"PLease extract and return both the patient's age and treatment from the following transcription. {transcription}"
        }
    ]

    function_definition = [{
        'type': 'function',
        'function': {
            'name': 'create_final_df',
            'description': 'This function formats the response, as a final DataFrame, the age, recommended treatment or procedure, medical specialty, and ICD code',
            'parameters': {
                'type': 'object',
                'properties': {
                    'Age': {'type': 'number', 'description':'THe age of the patient'},
                    'Recommended Treatment/Procedure': {'type': 'string', 'description': 'The recommended treatment/procedure or the patient\'s disease.'},
                }
            }
        }
    }]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=function_definition
    )
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

def get_icd_codes(treatment):
    if treatment != 'Unknown':
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"Provide the ICD codes for the following treatment or procedure: {treatment}. Return the answer as list of codes, please only include the code and no other information"
                }
            ],
            temperature=0
        )
        output = response.choices[0].message.content
    else:
        output = 'Unknown'
    return output

processed_data = []

for index, row in df.iterrows():
    medical_specialty = row['medical_specialty']
    extracted_data = extract_info_with_openai(row['transcription'])
    icd_code = get_icd_codes(extracted_data["Recommended Treatment/Procedure"]) if 'Recommended Treatment/Procedure' in extracted_data.keys() else 'Unknown'
    extracted_data['Medical Speciality'] = medical_specialty
    extracted_data['ICD Code'] = icd_code

    processed_data.append(extracted_data)

df_structured = pd.DataFrame(processed_data)
print(str(df_structured))
