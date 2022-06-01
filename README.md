A LYRICAL EVOLUTION: 

An Investigation of the Cultural Lexicon & Historical Relevance of U.S. Popular Music from 1958 - 2021

#### [Final Slide Presentation](________)
===
        
Team Members: Ben Smith, Chris Teceno, Jerry Nolf, Rachel Robbins-Mayhill  |   Codeup   |   Innis Cohort   |   June 2022
 
===


 <img src='Drafts/dataset-cover.png' width="1200" height="250" align="center"/>


===



Table of Contents
---
 
* I. [Project Overview](#i-project-overview)<br>
[1. Goal](#1-goal)<br>
[2. Description](#2-description)<br>
[3. Formulating Hypotheses](#3-formulating-hypotheses)<br>
[4. Initial Questions](#4initial-questions)<br>
[5. Key Findings](#5-key-findings)<br>
[6. Deliverables](#6-deliverables)<br>
* II. [Project Data Context](#ii-project-data-context)<br>
[1. Data Dictionary](#1-data-dictionary)<br>
* III. [Project Plan - Data Science Pipeline](#iii-project-plan---using-the-data-science-pipeline)<br>
[1. Project Planning](#1-plan)<br>
[2. Data Acquisition](#2-acquire)<br>
[3. Data Preparation](#3-prepare)<br>
[4. Data Exploration](#4explore)<br>
[5. Modeling & Evaluation](#5-model--evaluate)<br>
[6. Product Delivery](#6-delivery)<br>
* IV. [Project Modules](#iv-project-modules)<br>
* V. [Project Reproduction](#v-project-reproduction)<br>
 
 
 
## I. PROJECT OVERVIEW

#### 1. GOAL:
The goal of this project is to build a Natural Language Processing model that can predict the decade a song first appeared on the Billboard Top 100 chart, given the lyrics of the song. 
 
 
#### 2. DESCRIPTION:
This project was initiated by utilizing web scraping techniques to scrape song lyrics from Genius ______. ___________, were used as the documents within the corpus for this NLP project.

After acquiring and preparing the corpus, our team conducted natural language processing exploration utilizing methods such as word clouds and bigrams. We employed multiclass classification methods to create multiple machine learning models. The end goal was to create an NLP model that accurately predicted the decade a song first appeared on the Billboard Top 100 chart, based on the words and word combinations found in the lyrics of the song.
 

#### 3. FORMULATING HYPOTHESES
The initial hypothesis of the project was ______ would have a higher average word count in comparison to other _______. The thought behind this was _________ . These thoughts and the subsequent hypothesis drove the initial exploratory questions for this project.


#### 4.INITIAL QUESTIONS:
The focus of the project is on identifying the decade a song first appeared on the Billboard Top 100. Below are some of the initial questions this project looks to answer throughout the Data Science Pipeline.
 
 
##### Data-Focused Questions
- What are the most frequently occuring words?
- What are the most frequently occuring bigrams (pairs of words) by each decade?
- What decade did the song first appear in the top 100?
- What topics are most unique to each decade?
- Is there a correlation between sentiment and decade?
 
  
#### 5. KEY FINDINGS:
The key findings for this presentation are available in slide format by clicking on the [Final Slide Presentation](________________).

Ultimately, our hypothesis that  

Exploration revealed t

A lot of feature engineering was completed prior to modeling in an attempt to create a higher performing model, however our best performing models were made with 
 
#### 6. DELIVERABLES:
- [x] README file - provides an overview of the project and steps for project reproduction
- [x] Draft Jupyter Notebook - provides all steps taken to produce the project
- [x] .py modules - provide reproducible code to automate acquiring, preparing, splitting, exploring, and modeling the corpus
- [x] Final Jupyter Notebook - provides presentation-ready wrangle, exploration, modeling, and summary
- [x] Slide Deck - includes ___________
- [x] One-Page Handout Pamphlet - provides overview and summary of project process and outcome
- [x] 10 Minute Presentation 
 
 
## II. PROJECT DATA CONTEXT
 
#### 1. DATA DICTIONARY:
The final DataFrame used to explore the corpus for this project contains the following variables (columns).  The variables, along with their data types, are defined below:
 
 
|  Variables             |    Definition                              |    DataType             |
| :--------------------   | :---------------------------------------- | :-------------------- |
deade (target variable) | The decade the song was FIRST listed on the Billboard Top 100 | object
original              | The contents of scraped song lyrics      | object
more_clean*           | The lyric contents cleaned with prep_data function     | object
unique_words*         | The unique words in each more_clean document            | object
char_count*           | The number of characters within the more_clean document | object
word_count*           | The number of words within the more_clean document      | object
unique_word_count*    | The number of unique words in the more_clean document   | object
most_common_word*     | The word that occured most within the document          | object
2nd_most_common_word* | The 2nd most commonly occuring word within the document | object
3rd_most_common_word* | The 3rd most commonly occuring word within the document | object
4th_most_common_word* | The 4th most commonly occuring word within the document | object
5th_most_common_word* | The 5th most commonly occuring word within the document | object
 
* feature engineered
 
## III. PROJECT PLAN - USING THE DATA SCIENCE PIPELINE:
The following outlines the process taken through the Data Science Pipeline to complete this project. 
 
Plan➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver
 
#### 1. PLAN
- [x]  Review project expectations
- [x]  Draft project goal to include measures of success
- [x]  Create questions related to the project
- [x]  Create questions related to the corpus
- [x]  Create a plan for completing the project using the data science pipeline
- [x]  Create a data dictionary framework to define variables and data context
- [x]  Draft starting hypothesis
 
#### 2. ACQUIRE
- [x]  Create .gitignore
- [x]  Create env file with log-in credentials for GitHub
- [x]  Store env file in .gitignore to ensure the security of sensitive data
- [x]  Create wrangle.py module
- [x]  Store functions needed to acquire the lyric documents for __________
     - [x] 
- [x]  Ensure all imports needed to run the acquire functions are inside the wrangle.py document
- [x]  Using Command Line / Terminal, run ‘python acquire.py’ to create the `data.json` file that contains the corpus
- [x]  Using Jupyter Notebook or other Python Scripting Program
     - [x]  Run all required imports
     - [x]  Import functions for acquiring the corpus from wrangle.py module
     - [x]  Obtain the original size of the corpus
     
 
#### 3. PREPARE
Using Jupyter Notebook
- [x]  Acquire corpus using functions from the wrangle.py module
- [x]  Summarize corpus using methods and document observations
- [x]  Clean documents/corpus
   - [x] Make all text lowercase
   - [x] Normalize, encode, and decode to remove accented text and special characters
   - [x] Tokenize strings to break words and punctuation into discrete units
   - [x] Stem or Lemmatize words to acquire base words
   - [x] Remove stopwords
- [x]  Address missing values, data errors, unnecessary data, renaming
- [x]  Split corpus into train, validate, and test samples
Using Python Scripting Program (Jupyter Notebook)
- [x]  Create prepare functions within prepare.py
- [x]  Store functions needed to prepare the ________ Corpus such as:
   - [x]  Cleaning Function: to normalize text and remove accented and special characters
   - [x]  Tokenize Function: to break strings down into discrete units
   - [x]  Stem Function: to acquire root words
   - [x]  Lemmatize Function: to acquire lexicographically correct root words
   - [x]  Stopwords Function: to remove meaningless words
- [x]  Ensure all imports needed to run the prepare functions are added to the prepare.py document

 
#### 4.EXPLORE
Using Jupyter Notebook:
- [x]  Answer key questions about hypotheses 
- [x]  Create visualizations with the intent to discover variable relationships
     - [x]  Identify variables related to programming langauge
     - [x]  Identify any potential data integrity issues
- [x]  Document findings
- [x]  Summarize conclusions, provide clear answers, and summarize takeaways
     - [x] Explain plan of action as deduced from work to this point    
- [x]  Create prepare functions within explore.py
- [x]  Store functions needed to explore the ________ Corpus in explore.py
- [x]  Ensure all imports needed to run the explore functions are added to the explore.py document
 
#### 5. MODEL & EVALUATE
Using Jupyter Notebook:
- [x]  Establish baseline accuracy
- [x]  Create two types of dataframes to run through models (TF-IDF, and Feature Engineering)
- [x]  Train and fit multiple (1000+) models with varying algorithms and/or hyperparameters
- [x]  Compare evaluation metrics across models
- [x]  Remove unnecessary features
- [x]  Evaluate best performing models using validate set
- [x]  Choose best performing validation model for use on test set
- [x]  Test final model on out-of-sample testing corpus
- [x]  Summarize performance
- [x]  Interpret and document findings
- [x]  Create prepare functions within model.py
- [x]  Store functions needed to model the __________ Corpus in model.py
- [x]  Ensure all imports needed to run the model functions are added to the model.py document
 

#### 6. DELIVERY
- [x]  Prepare a presentation using Google Slides, containing 2-5 slides
     - [x]  Include an introduction of the project and goals
     - [x]  Provide an executive summary of findings, key takeaways, recommendations, and next steps
     - [x]  Create a walkthrough of the exploration analysis 
     - [x]  Include 2+ presentation-worthy visualizations that support exploration and modeling
     - [x]  Provide final takeaways, recommend a course of action for modeling, and next steps
- [x]  Prepare final notebook in Jupyter Notebook
     - [x]  Create clear walk-though of the Data Science Pipeline using headings and dividers
     - [x]  Explicitly define questions asked during the initial analysis
     - [x]  Visualize relationships
     - [x]  Document takeaways
     - [x]  Comment code thoroughly



 
 
## IV. PROJECT MODULES:
- [x] wrangle.py - provides reproducible python code to automate acquisition 
- [x] prepare.py - provides reproducible python code to automate cleaning, preparing, and splitting the corpus
- [x] explore.py - provides reproducible python code to automate exploration and visualization
- [x] model.py - provides reproducible python code to automate modeling and evaluation
  
## V. PROJECT REPRODUCTION:
### Steps to Reproduce
- [x] Create an env.py file that contains access credentials to ________ in the form of ________ and ________ variables
- [x] You will need a ________ personal access token in place of a password in order to access the contents within _________
     - [x] Go here and generate a personal access token: _________
     - [x] You do _not_ need to select any scopes, i.e. leave all the checkboxes unchecked
     - [x] Save the token in your env.py file under the variable `________`
- [x] Add your _______ username to your env.py file under the variable `________`
- [x] Store that env file locally in the repository
- [x] Make .gitignore and confirm .gitignore is hiding your env.py file
- [x] Clone our repo (including the wrangle.py)
- [x] Import python libraries:  pandas, matplotlib, seaborn, numpy, sklearn, nltk, json, re, and unicodedata
- [x] Follow steps as outlined in the README.md. and work.ipynb
- [x] Run Final_Report.ipynb to view the final product
