# Autism Screening Adult Machine Learning Project

## Project Overview
This project will implement a machine learning model using C++20 to screen for autism based on the Autism Screening Adult (UCI) dataset. The model aims to predict ASD classification from questionnaire scores and demographic features.

## Dataset Description
- Dataset: Autism Screening Adult (UCI)
- Source: https://archive.ics.uci.edu/dataset/426/autism+screening+adult
- Records: 704
- Features: 20
- Target: Class/ASD (screening classification, not confirmed diagnosis)

## Feature Table
| Column | Type | Description |
|--------|------|-------------|
| A1_Score | Categorical | Binary AQ-10 questionnaire score |
| A2_Score | Categorical | Binary AQ-10 questionnaire score |
| A3_Score | Categorical | Binary AQ-10 questionnaire score |
| A4_Score | Categorical | Binary AQ-10 questionnaire score |
| A5_Score | Categorical | Binary AQ-10 questionnaire score |
| A6_Score | Categorical | Binary AQ-10 questionnaire score |
| A7_Score | Categorical | Binary AQ-10 questionnaire score |
| A8_Score | Categorical | Binary AQ-10 questionnaire score |
| A9_Score | Categorical | Binary AQ-10 questionnaire score |
| A10_Score | Categorical | Binary AQ-10 questionnaire score |
| age | Numerical | Age of the individual |
| gender | Categorical | Gender of the individual |
| ethnicity | Categorical | Ethnicity of the individual |
| jundice | Categorical | Presence of jaundice |
| austim | Categorical | Family history of autism |
| contry_of_res | Categorical | Country of residence |
| used_app_before | Categorical | Previous app usage |
| result | Numerical | Total AQ-10 screening score |
| age_desc | Categorical | Age description |
| relation | Categorical | Relationship |
| Class/ASD | Target | Screening classification |

## Missing Values
- age: 2 missing values
- ethnicity: 95 missing values
- relation: 95 missing values

## Feature Type Detection Rule
- Non-numeric features are categorical
- Numeric features with at most 2 unique values are categorical
- All other numeric features are numerical
- Missing values are represented by '?' and are ignored during type detection