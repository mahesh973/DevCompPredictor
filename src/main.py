#Importing the necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle

from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate
from collections import Counter

from plotting_helpers import survey_responses_count, median_compensation, dev_type, industry_salaries, years_WorkExp, median_salary_Ed_Level
from plotting_helpers import languages_worked_with, databases_worked_with
from helpers import salary_ranges, slider_ranges, perform_encoding
from helpers import OrgSize_keys, EdLevel_keys, Industry_keys, DevType_keys, Age_keys, RemoteWork_keys, IcorPM_keys, Employment_keys


# Set API Key and initialize agents
API_KEY = st.secrets["OPENAI_API_KEY"]
df = pd.read_csv("../stack-overflow-developer-survey-results-2023/survey_results_public.csv")
chat = ChatOpenAI(model_name='gpt-4-0613', temperature=0.2, api_key=API_KEY)
agent = create_pandas_dataframe_agent(chat, df, verbose=True)

# Custom Headers for enhancing UI Text elements
def custom_header(text, level=1):
    if level == 1:
        icon_url = "https://stackoverflow.com/favicon.ico"
        # Adjust the img style as needed (e.g., height, vertical alignment, margin)
        st.markdown(f"""
            <h1 style='text-align: center;'>
                <img src="{icon_url}" alt="Icon" style="vertical-align: middle; height: 42px; margin-right: 10px;">
                <span style='color: #F48024; font-family: sans serif;'>{text}</span>
            </h1>
        """, unsafe_allow_html=True)
        #st.markdown(f"<h1 style='text-align: center; color: #ef8236; font-family: sans serif;'>{text}</h1>", unsafe_allow_html=True)
    elif level == 2:
        st.markdown(f"<h2 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h2>", unsafe_allow_html=True)
    elif level == 3:
        st.markdown(f"<h3 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h3>", unsafe_allow_html=True)
    elif level == 4:
        st.markdown(f"<h5 style='text-align: center; color: #00749C; font-family: sans serif;'>{text}</h5>", unsafe_allow_html=True)
    elif level == 5:
        st.markdown(f"<h5 style='text-align: center; color: #f63366; font-family: sans serif;'>{text}</h5>", unsafe_allow_html=True)

# Helper function for predicting compensation
def predict_compensation(features):
    with open("../saved_weights/salary_mapping.pkl", 'rb') as f:
        salary_mapping = pickle.load(f)
    
    with open("../saved_weights/hybrid_classifier.pkl", 'rb') as file:
        hybrid_classifier = pickle.load(file)
    
    encoded_response = hybrid_classifier.predict(features)

    target_category = next((k for k, v in salary_mapping.items() if v == encoded_response[0]), None)
    
    return target_category

# Helper function to display key insights
def plot_eda_charts(level):
    if level == 1:
        sorted_df = survey_responses_count(df)

        fig = px.bar(sorted_df, x='count', y='Country', color='Country', labels={'count': 'Number of Survey Responses'}, orientation='h')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

        return fig

    if level == 2:
        salary_stats_df = median_compensation(df)

        fig = px.bar(salary_stats_df,  y=salary_stats_df.index, x='median', labels={'median': 'Median Compensation (USD)'},
                color='median', orientation='h', color_continuous_scale=px.colors.qualitative.Set1) 
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)

        return fig
    
    if level == 3:
        df_databases = databases_worked_with(df)

        all_dbs = [db for sublist in df_databases['DatabaseHaveWorkedWith'] for db in sublist]
        dbs_counts = Counter(all_dbs)

        # Sort databases by counts and select top 20
        top20_dbs_counts = dbs_counts.most_common(20)
        dbs, counts = zip(*top20_dbs_counts) 

        vibrant_colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
        ]

        fig = go.Figure(data=[go.Bar(y=dbs, x=counts, orientation='h', marker_color=vibrant_colors 
        )])

        fig.update_layout( xaxis_title="Usage Count", yaxis_title="Databases",
            yaxis={'categoryorder':'total ascending'}, height=500 
        )

        return fig
    
    if level == 4:
        df_languages = languages_worked_with(df)

        all_languages = [db for sublist in df_languages['LanguageHaveWorkedWith'] for db in sublist]
        languages_counts = Counter(all_languages)

        # Sort databases by counts and select top 20
        top20_languages_counts = languages_counts.most_common(20)
        languages, counts = zip(*top20_languages_counts)

        vibrant_colors = ['#17bebb', '#ff6f61','#6b5b95', '#88b04b', '#f7cac9', '#92a8d1', '#955251', '#b565a7', '#009B77',
                          '#DD4124', '#D65076', '#45B8AC', '#EFC050', '#5B5EA6', '#9B2335', '#e6194B', '#3cb44b', '#ffe119',
                          '#4363d8', '#f58231' ] 

        fig = go.Figure(data=[go.Bar(y=languages, x=counts, orientation='h', marker_color=vibrant_colors 
        )])

        fig.update_layout( xaxis_title="Usage Count", yaxis_title="Languages",
            yaxis={'categoryorder':'total ascending'}, height=500 
        )

        return fig
    
    if level == 5:
        dev_type_df = dev_type(df)

        fig = px.bar(dev_type_df, y='DevType', x='ConvertedCompYearly', color='DevType', 
                     labels={'ConvertedCompYearly': 'Median Salary (USD)', 'DevType': 'Developer Type'}, orientation='h',height=550,
                    color_continuous_scale=px.colors.qualitative.Set2)  

        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, xaxis_title='Median Salary (USD)',
                        margin=dict(l=20, r=20, t=30, b=20))
        return fig
    
    if level == 6:
        industry_df = industry_salaries(df)

        fig = px.bar(industry_df, y='Industry', x='ConvertedCompYearly', color='Industry',  
            labels={'ConvertedCompYearly': 'Median Salary (USD)', 'Industry': 'Industry'}, orientation='h',
            height = 500, color_continuous_scale=px.colors.qualitative.Set3)  

        fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
        return fig
    
    if level == 7:
        salary_stats = years_WorkExp(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=salary_stats['WorkExp'], y=salary_stats['median'], mode='lines+markers',
                                name='Median Salary', line=dict(color='royalblue')))

        lower_bound = np.maximum(salary_stats['median'] - salary_stats['std'], 0)

        fig.add_trace(go.Scatter(x=salary_stats['WorkExp'], y=salary_stats['median'] + salary_stats['std'],
                                mode='lines', name='Upper Bound', marker=dict(color="#444"),
                                line=dict(width=0), showlegend=False))

        fig.add_trace(go.Scatter(x=salary_stats['WorkExp'], y=lower_bound, mode='lines', name='Lower Bound',
                                marker=dict(color="#444"), line=dict(width=0),
                                fillcolor='rgba(68, 68, 68, 0.3)',fill='tonexty',showlegend=False))

        fig.update_layout(xaxis_title='Years of Work Experience',
                        yaxis_title='Median Salary (Yearly)',
                        hovermode="x", height = 400)
        
        return fig 
    
    if level == 8:
        median_salary_by_edlevel_sorted = median_salary_Ed_Level(df)

        fig = go.Figure()
        fig = px.bar(median_salary_by_edlevel_sorted, y='EdLevel', x='ConvertedCompYearly', color='EdLevel',  
                    labels={'ConvertedCompYearly': 'Median Salary (Yearly)', 'EdLevel': 'Education Level'},
                    orientation='h', height=400)  

        fig.update_layout(yaxis={'categoryorder':'total ascending'},  
                        coloraxis_colorbar=dict(title='Median Salary'))
        
        return fig 

# Main Streamlit UI code

st.set_page_config(page_title='StackOverflow Developer Survey Results 2023', page_icon='📋',
                    layout="wide", initial_sidebar_state='collapsed')

custom_header('Stackoverflow Developer Survey Results 2023',level=1)

# Introduction content

introduction_text = """
<div style='text-align: center; color: #333; font-size: 20px;'>
    <p>Discover the latest trends and insights from the world's largest developer survey.</p>
    <p>Explore the insights, play with predicting the compensation ranges, and interact with our chat feature for a deeper dive into the tech industry of 2023. </p> 
</div>
"""

st.markdown(introduction_text, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Key Insights", "Compensation Range Predictor", "Chatbot"])

headers = ["Number of Survey Responses According to the Country", "Median Compensation across Countries in USD",
           "Top 20 Widely Used Databases by Developers in 2023", "Top 20 Widely Used Programming Languages by Developers in 2023",
           "Median Salary by Developer Profession in USA", "Median Salary by Industry in USA",
           "Median Salary by Years of Work Experience in USA", "Median Salary by Education Level in USA"]

with tab1:
    for i in range(0, len(headers), 2):
        cols = st.columns(2)  # Create two columns
        
        with cols[0]:
            custom_header(headers[i], level=4)
            fig = plot_eda_charts(level=i+1)
            st.plotly_chart(fig, use_container_width=True)
        
        if (i+1) < len(headers):
            with cols[1]:
                custom_header(headers[i+1], level=4)
                fig = plot_eda_charts(level=i+2)
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    custom_header("Compensation Range Predictor",level = 2)
    st.write("\n")

    ypc = st.slider('Years of professional coding experience',0, 50, 3)

    col1, spacer, col2 = st.columns([5, 2, 5])  
    
    with col1:
        organizational_size = st.selectbox("Organizational Size", OrgSize_keys, key = 'organizational_size_2')
        industry = st.selectbox("Industry", Industry_keys, key = 'industry_2')
        dev_type = st.selectbox("DevType", DevType_keys, key = 'dev_type_2')
        icor_pm = st.selectbox("IcorPM", IcorPM_keys, key='IcorPM_2')
    
    with col2:
        ed_level = st.selectbox("Education Level", EdLevel_keys, key='ed_level_2')
        age = st.selectbox("Age", Age_keys, key='age_2')
        remote_work = st.selectbox("WorkMode", RemoteWork_keys, key='remote_work_2')
        employment = st.selectbox("Employment", Employment_keys, key='employment_2')
    
    st.write("\n")

    if st.button("Predict Compensation"):
        features = {'Age' : [age],
                    'Employment' : [employment],
                    'RemoteWork' : [remote_work],
                    'EdLevel' : [ed_level],
                    'YearsCodePro' : [ypc] if ypc!=0 else [0.5],
                    'DevType' : [dev_type], 
                    'Industry' : [industry],
                    'OrgSize' : [organizational_size], 
                    'ICorPM' : [icor_pm]
                    }
        
        encoded_features = perform_encoding(features)

        prediction = predict_compensation(encoded_features)

        st.markdown(f"<div style='background-color:#A8E6CF;padding:10px;border-radius:10px;font-size:20px;'>"
            f"Predicted Compensation Range: <b>{salary_ranges[prediction]} USD</b></div>", unsafe_allow_html=True)
        
        st.write("\n")

        compensation_slider = st.slider("Compensation Range ($)",
                                min_value=min(slider_ranges.values())[0],  # Minimum of all ranges
                                max_value=max(slider_ranges.values())[1],  # Maximum of all ranges
                                value=slider_ranges[prediction],
                                step=500, format="%d$")

with tab3:
    custom_header("Chat Feature to ask Questions about the Survey Data",level=3)
    st.write("\n")
    
    # Sidebar with predefined questions
    predefined_questions = ["What are the most popular programming languages in 2023 according to the StackOverflow survey?",
    "Which cloud platforms were mostly used by developers in 2023?",  
    "What are the most common educational backgrounds for developers in 2023?",
    "Which profession is rewarding to choose for a person having an I.T. background?",
    "What was the median salary of Data Scientists in the IT field in USA?",
    "What was the median annual salary for Data Scientists in the IT sector in the United States, including a breakdown by Education level?"]  # Replace with your actual questions
    
    selected_question = st.sidebar.selectbox("Choose a Query", [""] + predefined_questions)
    # Text area for query input
    query = st.text_area("Enter your query:", value=selected_question if selected_question else "", placeholder="Which cloud platforms were most used by developers in 2023?")

    # Generate response
    if st.button("Get Response"):
        if query:
            with st.spinner("Generating response...."):
                try:
                    prompt_template = """Given the following dataframe and a question, generate an answer only based on the passed dataframe. 
                    In case if you're unable to find it, go to this link "https://survey.stackoverflow.co/2023/" to figure out the answer.
                    If the answer is not found, kindly state "Sorry, I don't know." Don't try to make up an answer.
                    QUERY: {query}"""

                    PROMPT = ChatPromptTemplate.from_template(
                        template=prompt_template
                    )
                    final_prompt = PROMPT.format_messages(query = query)
                    response = agent.invoke(final_prompt)
                    st.write("\n")
                    st.write(response['output'])
                except:
                    st.error("Please try again :(")
        else:
            st.warning("Please enter the query!")
    
    for i in range(12):
        st.write("\n")
    
    st.caption("Tip : Feel free to utilize the sidebar ('>') for query thoughts.")