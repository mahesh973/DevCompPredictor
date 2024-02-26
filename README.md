## StackOverflow Insights 2023

![Python version](https://img.shields.io/badge/python-3.9+-blue.svg) ![Langchain version](https://img.shields.io/badge/langchain-0.1.4-lightgreen.svg) ![Streamlit version](https://img.shields.io/badge/streamlit-1.31.1-red.svg)

## Description

StackOverflow Insights 2023 is an application designed to provide users with in-depth analysis and interactive exploration of the StackOverflow Developer Survey Results 2023. Our application is built for developers, data scientists, and anyone interested in the software development industry's trends and insights. It leverages the comprehensive StackOverflow Developer Survey Results to provide actionable insights and foster a deeper understanding of the industry landscape.

 This application offers three tabs, each offering unique functionalities:

1. **Key Insights**: Presents major findings and trends from the survey data through visually appealing charts and graphs.
2. **Compensation Range Predictor**: Allows users to estimate their potential salary range based on their role, years of coding experience, Industry, and other relevant factors.
3. **Chat Feature**: Offers an interactive chat interface where users can ask specific questions about the survey data.

## App Demo

![Demo](https://github.com/mahesh973/DevCompPredictor-Capstone/assets/59694546/35a69e20-ece6-47c9-b755-4d49c15f4970)

&nbsp;

## Project Workflow

![System Architecture](https://github.com/mahesh973/DevCompPredictor-Capstone/assets/59694546/0bee2a83-3db3-4400-b5f6-6c10bcea9274)

## Installation

### Dependencies

Before installing the application, ensure you have the following key dependencies installed on your system:

- Python 3.9 or higher
- Pandas
- Plotly
- Matplotlib
- Scikit-learn
- Streamlit
- Langchain
  
&nbsp;

**Note:** Make sure to have an `OPENAI_API_KEY` to access the chat feature. 

## User installation

**1. Clone the Repository**
```python
git clone "https://github.com/mahesh973/DevCompPredictor-Capstone"
```
**2. Navigate to the Directory**
```python
cd DevCompPredictor-Capstone
```
**3. Install the necessary dependencies**
```python
pip install -r requirements.txt
```
**4. Launch the application**
```python
streamlit run src/main.py
```

After completing these steps, the application should be running on your local server. Open your web browser and navigate to http://localhost:8501 to start exploring the StackOverflow Developer Survey Results 2023.


## References

This application is built using Streamlit. For more detailed information to explore the raw data, visit the official StackOverflow survey page:

[StackOverflow Developer Survey 2023](https://insights.stackoverflow.com/survey)

Feel free to contribute to the project by submitting issues or pull requests on GitHub. Your feedback and contributions are highly appreciated!

