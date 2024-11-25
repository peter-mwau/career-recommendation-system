# Web Application Page
# TMI4033 Project: IT-Career Recommendation System with RF-machine-learning-model

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import sklearn
from datasets import load_dataset

data = pd.read_csv("Dataset/mldata.csv")


# load prediction model from notebook
pickleFile = open('rfweights.pkl', 'rb')
rfmodel = pickle.load(pickleFile)

# Obtain the categorical/nominal data because it is not coded according (but based on the first occurence, first come first assign number)
# Therefore, need to read from the file to obtain the number.
categorical_cols = data[[
    'certifications',
    'workshops',
    'Interested subjects',
    'interested career area ',
    'Type of company want to settle in?',
    'Interested Type of Books'
]]
# assign the datatype and automated assigned code
for i in categorical_cols:
    data[i] = data[i].astype('category')
    data[i] = data[i].cat.codes

# embedded nominal/ categorical values for certicates
certificates_name = list(categorical_cols['certifications'].unique())
certificates_code = list(data['certifications'].unique())
certificates_references = dict(zip(certificates_name, certificates_code))

# embedding for workshops
workshop_name = list(categorical_cols['workshops'].unique())
workshop_code = list(data['workshops'].unique())
workshop_references = dict(zip(workshop_name, workshop_code))

# embedding for subjects_interests
subjects_interest_name = list(categorical_cols['Interested subjects'].unique())
subjects_interest_code = list(data['Interested subjects'].unique())
subjects_interest_references = dict(
    zip(subjects_interest_name, subjects_interest_code))

# embedding for career_interests
career_interest_name = list(
    categorical_cols['interested career area '].unique())
career_interest_code = list(data['interested career area '].unique())
career_interest_references = dict(
    zip(career_interest_name, career_interest_code))

# embedding for company_intends
company_intends_name = list(
    categorical_cols['Type of company want to settle in?'].unique())
company_intends_code = list(
    data['Type of company want to settle in?'].unique())
company_intends_references = dict(
    zip(company_intends_name, company_intends_code))

# embedding for book_interests
book_interest_name = list(
    categorical_cols['Interested Type of Books'].unique())
book_interest_code = list(data['Interested Type of Books'].unique())
book_interest_references = dict(zip(book_interest_name, book_interest_code))


def greet(name):
    return f"Hello, {name}!"


'''#dummy encode
def dummy_encode(df):
    if input == "Management":
        return [1, 0]
    elif input == "Technical":
        return [0, 1]
    elif input == "smart worker":
        return [1, 0]
    elif input == "hard worker":
        return [0, 1]
    else:
        return "Invalid choice"'''


def rfprediction(name, logical_thinking, hackathon_attend, coding_skills, public_speaking_skills,
                 self_learning, extra_course, certificate_code, worskhop_code, read_writing_skill, memory_capability, subject_interest, career_interest, company_intend, senior_elder_advise, book_interest, introvert_extro,
                 team_player, management_technical, smart_hardworker):
    df = pd.DataFrame.from_dict(
        {
            "logical_thinking": [logical_thinking],
            "hackathon_attend": [hackathon_attend],
            "coding_skills": [coding_skills],
            "public_speaking_skills": [public_speaking_skills],
            "self_learning": [self_learning],
            "extra_course": [extra_course],
            "certificate": [certificate_code],
            "workshop": [worskhop_code],
            "read_writing_skills": [
                (0 if "poor" in read_writing_skill else 1 if "medium" in read_writing_skill else 2)
            ],
            "memory_capability": [
                (0 if "poor" in memory_capability else 1 if "medium" in memory_capability else 2)
            ],
            "subject_interest": [subject_interest],
            "career_interest": [career_interest],
            "company_intend": [company_intend],
            "senior_elder_advise": [senior_elder_advise],
            "book_interest": [book_interest],
            "introvert_extro": [introvert_extro],
            "team_player": [team_player],
            "management_technical": [management_technical],
            "smart_hardworker": [smart_hardworker]
        }
    )

    # replace str to numeric representation, dtype chged to int8
    df = df.replace({"certificate": certificates_references,
                     "workshop": workshop_references,
                     "subject_interest": subjects_interest_references,
                     "career_interest": career_interest_references,
                     "company_intend": company_intends_references,
                     "book_interest": book_interest_references})

    # dummy encoding
    # first we convert into list from df
    userdata_list = df.values.tolist()
    # now we append boolean based conditions
    if (df["management_technical"].values == "Management"):
        userdata_list[0].extend([1])
        userdata_list[0].extend([0])
        userdata_list[0].remove('Management')
    elif (df["management_technical"].values == "Technical"):
        userdata_list[0].extend([0])
        userdata_list[0].extend([1])
        userdata_list[0].remove('Technical')
    else:
        return "Err"

    if (df["smart_hardworker"].values == "smart worker"):
        userdata_list[0].extend([1])
        userdata_list[0].extend([0])
        userdata_list[0].remove('smart worker')
    elif (df["smart_hardworker"].values == "hard worker"):
        userdata_list[0].extend([0])
        userdata_list[0].extend([1])
        userdata_list[0].remove('hard worker')
    else:
        return "Err"

    prediction_result = rfmodel.predict(userdata_list)
    prediction_result_all = rfmodel.predict_proba(userdata_list)
    print(prediction_result_all)
    # create a list for output
    result_list = {"Applications Developer": float(prediction_result_all[0][0]),
                   "CRM Technical Developer": float(prediction_result_all[0][1]),
                   "Database Developer": float(prediction_result_all[0][2]),
                   "Mobile Applications Developer": float(prediction_result_all[0][3]),
                   "Network Security Engineer": float(prediction_result_all[0][4]),
                   "Software Developer": float(prediction_result_all[0][5]),
                   "Software Engineer": float(prediction_result_all[0][6]),
                   "Software Quality Assurance (QA)/ Testing": float(prediction_result_all[0][7]),
                   "Systems Security Administrator": float(prediction_result_all[0][8]),
                   "Technical Support": float(prediction_result_all[0][9]),
                   "UX Designer": float(prediction_result_all[0][10]),
                   "Web Developer": float(prediction_result_all[0][11]),
                   }
    return result_list


cert_list = ["app development", "distro making", "full stack", "hadoop",
             "information security", "machine learning", "python", "r programming", "shell programming"]
workshop_list = ["cloud computing", "data science", "database security",
                 "game development", "hacking", "system designing", "testing", "web technologies"]
# can be used in this section and memory capability section
skill = ["excellent", "medium", "poor"]
subject_list = ["cloud computing", "Computer Architecture", "data engineering", "hacking",
                "IOT", "Management", "networks", "parallel computing", "programming", "Software Engineering"]
career_list = ["Business process analyst", "cloud computing",
               "developer", "security", "system developer", "testing"]
company_list = ["BPA", "Cloud Services", "Finance", "Product based", "product development", "SAaS services",
                "Sales and Marketing", "Service Based", "Testing and Maintainance Services", "Web Services"]
book_list = ["Action and Adventure", "Anthology", "Art", "Autobiographies", "Biographies", "Childrens", "Comics", "Cookbooks", "Diaries", "Dictionaries", "Drama", "Encyclopedias", "Fantasy", "Guide",
             "Health", "History", "Horror", "Journals", "Math", "Mystery", "Poetry", "Prayer books", "Religion-Spirituality", "Romance", "Satire", "Science", "Science fiction", "Self help", "Series", "Travel", "Trilogy"]
Choice_list = ["Management", "Technical"]
worker_list = ["hard worker", "smart worker"]

demo = gr.Interface(fn=rfprediction, inputs=[
    gr.Textbox(placeholder="What is your name?", label="Name"),
    gr.Slider(minimum=1, maximum=9, value=3, step=1,
              label="Are you a logical thinking person?", info="Scale: 1 - 9"),
    gr.Slider(minimum=0, maximum=6, value=0, step=1, label="Do you attend any Hackathons?",
              info="Scale: 0 - 6 | 0 - if not attended any"),
    gr.Slider(minimum=1, maximum=9, value=5, step=1,
              label="How do you rate your coding skills?", info="Scale: 1 - 9"),
    gr.Slider(minimum=1, maximum=9, value=3, step=1,
              label="How do you rate your public speaking skills/confidency?", info="Scale: 1 - 9"),
    gr.Radio({"Yes", "No"}, type="index",
             label="Are you a self-learning person? *"),
    gr.Radio({"Yes", "No"}, type="index",
             label="Do you take extra courses in uni (other than IT)? *"),
    gr.Dropdown(cert_list, label="Select a certificate you took!"),
    gr.Dropdown(workshop_list, label="Select a workshop you attended!"),
    gr.Dropdown(skill, label="Select your read and writing skill"),
    gr.Dropdown(skill, label="Is your memory capability good?"),
    gr.Dropdown(subject_list, label="What subject you are interested in?"),
    gr.Dropdown(career_list, label="Which IT-Career do you have interests in?"),
    gr.Dropdown(
        company_list, label="Do you have any interested company that you intend to settle in?"),
    gr.Radio({"Yes", "No"}, type="index",
             label="Do you ever seek any advices from senior or elders? *"),
    gr.Dropdown(book_list, label="Select your interested genre of book!"),
    gr.Radio({"Yes", "No"}, type="index",
             label="Are you an Introvert?| No - extrovert *"),
    gr.Radio({"Yes", "No"}, type="index", label="Ever worked in a team? *"),
    gr.Dropdown(
        Choice_list, label="Which area do you prefer: Management or Technical?"),
    gr.Dropdown(worker_list, label="Are you a Smart worker or Hard worker?")
],
    outputs=gr.Label(num_top_classes=5),
    title="IT-Career Recommendation System: TMI4033 Colletive Intelligence",
    description="Members: Derrick Lim Kin Yeap 74597, Jason Jong Sheng Tat 75125, Jason Ng Yong Xing 75127, Muhamad Hazrie Bin Suhkery 73555 "
)


# main
if __name__ == "__main__":
    demo.launch(share=True)
