# Introduction
Dived into Sentiment Analysis and Topic Modeling using Natural Language Processing & Deep Learning focusing on extracting and analyzing opinions expressed from comments scraped from a YouTube, Data Cleaning & Preprocssing of Natural Language datasr, Text Analysis to visually explore and Identify recurring themes or topics present in the comments, explored Natural Language Technique of TF-IDF + Logistic regression by buildig a baseline based on Tf-Idf representations of product reviews and finally Topic Modeling using deep learning technique of Bidirectional Encoder Representations from Transformers (BERT) to identify recurring topics within the comments

Python code? Check them out here on GitHub: https://bit.ly/4c5RBUB

# Background
Driven by the quest to navigate the data analytics upskilling by chosing the best online courses, this project was born bt a desire to pinpoint top skills offered by the Google Data Analytics Certifcate helping others to find optimal jobs.

The data was scraped from comments from a YouTube Video posted by **Luke Barousse** titled "Become a DATA ANALYST with NO degree?!? The Google Data Analytics Professional Certificate", Link to the video on YouTube: https://www.youtube.com/watch?v=fmLPS6FBbac. The video is packed with overview about the course, tools taught in the course, likes and concerns towards the course and some recomendations

### The five questions I wanted to answer through this project were;

1. What are the overall sentiment expressed in the comments towards the course?
2. What are the recurring themes (revalent areas of interest, concerns, or experiences) shared by learners through the comments?
3. How can Logistic Regression with TF-IDF vectorization be utilized for text classification tasks?
4. How can topic modeling using the deep learning technique of BERT (Bidirectional Encoder Representations from Transformers) be applied to identify recurring themes or topics within comments. 
5. What insights can be gained from identifying these recurring themes or topics, and how can this information be utilized to improve the course?

# Tools I Used
For my machine learning project sentiment analysis & topic modeling, I hannesed the powert of several key tools:

- **YouTube Data API** For extracting comments from YouTube Video for my analysis and machine learning modeling
- **Python:** My go-to programing language for data cleaning, preprocessing, exploratory data analysis and implementing machine learning algorithm
- **Logistic Regression with TF-IDF vectorizer:** As the predictive model generarting input features using the text classification and sentiment analysis 
- **BERT** (Bidirectional Encoder Representations from Transformers), a deep learning model for topic modeling
- **Gradio:** An open-source Python package to build a web application for the machine learning model. You can then share a link to your demo or web application in just a few seconds using Gradio's built-in sharing features.
- **Git and GitHub:** Essentials for version control and sharing my Python scripts and analysis ensuring colaboration and project tracking.

# The Analysis

The analysis for this project aimed investigating the overall sentiment expressed in the comments towards the course and identifying the recurring themes and topics shared by learners through the comments, and gain insights into key skills and career projections and recomendations.

### 1. What are the overall sentiment expressed in the comments towards the course?
To identify the overall sentiments expressed in comments towards the course, I applied the rule-based sentiment analysis using VADER lexicon, a rule-based sentiment analysis tool and analyzed the sentiment by extracting sentiment scores and classifying comments as positive or negative based on compound scores.

I then perfoemed a Text Categorization/Token Labbeling. This process enabled me to categorize of YouTube comments into positive and negative sentiments, aiding in sentiment analysis and understanding the feedback in the video conten.

The majority of the sentiments in the dataset are Positive (93.4%), while a small fraction is Negative (6.6%). This suggests that the overall sentiment is highly positive.

![Sentiment Scores](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/sentiment%20scores.png)

*Bar graph visualizing the sentiments scores*

### 2. What are the recurring themes: revalent areas of interest, skills, tools and experinces
Major tools used in the course included excel,sql, tableau and programing language python. The career projection included mentions career switch and change to data jobs after completing the course.

![Word Cloud](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/Word%20Cloud.png)

*Word Cloud showing recuring themes. The bigger the word appears, the more often it's mentioned within the video comment*

### 3. How can Logistic Regression with TF-IDF vectorization be utilized for text classification tasks?
The model perfoemed very well in identifying positive instances with high precision and perfect recall. However, due to the bias towards positive cases, it might be worth investigating the data distribution to see if there's a class imbalance issue.
- High Precision (0.99): Out of all positive predictions made by the model, 98.44% were actually correct. This means the model rarely makes mistakes when classifying a positive instance.
- Perfect Recall (1.0): The model identified all the positive instances correctly. There are no false negatives (missing positive cases).
- High F1 Score (0.99): The F1 score is a harmonic mean between precision and recall, and it takes into account both metrics. A score of 0.992 indicates that the model performs well overall.
- Looking closer at the individual metrics, it's interesting to note that the model has 100% recall but 0% true negatives. This suggests the model is biased towards predicting positive cases. This is be due to the class imbalance in the data, where there are significantly fewer negative instances compared to positive ones.

![Confusion Matrix](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/Confusion%20Matrix.png)

*Confusion Matrix showing the model performance*

### 4. How can topic modeling using the deep learning technique of BERT (Bidirectional Encoder Representations from Transformers) be applied to identify recurring themes or topics within comments?

**Training:** 
- I started by instantiating BERTopic and setting the language to english since the comment are in the English language.I then calculated the topic probabilities. However, this can slow down BERTopic significantly at large amounts of data (>100_000 documents). It is advised to turn this off if you want to speed up the model.

**Topic Extraction and Representation:**
 - I extracted a total of 39 topics in descending order of topic size/count. the counts indicates the number of words in each topic while signifies the name assigned to each topic.For each topic, we can access the top words along with their corresponding c-TF-IDF score. A higher score denotes greater relevance of the word in representing the topic.

**Topics Visualization:** 
- Topic 1: Keywords: per, day, 10, hours, week
Insight: This topic seems related to time management or scheduling, with emphasis on daily or weekly hours, suggesting discussions about the time required for certain activities or commitments.

- Topic 2: Keywords: analyst, data, job, scientist
Insight: This topic centers on job roles in data analysis or data science, likely discussing careers, job responsibilities, or the nature of work in these fields.

- Topic 4: Keywords: python, programming, language, use, choose
Insight: This topic focuses on programming, specifically the Python that is the main programing language used in the course.

- Topic 7: Keywords: excel, SQL, tableau, teach, BI
Insight: This topic revolves around technical skills and tools, specifically Excel, SQL, Tableau, and business intelligence (BI), indicating a focus on teaching or using these tools.


![Topoic Word Scores](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/Topic%20Word%20Scores.png)

*Bar charts visualizing the most relevant words for the 8 topics*

**Intertopic Distance Map**
- I used BERTopic to implement an interactive dashboard showing for each topic the corresponding words and their scor using its visualize_topics() function and even go one step further by giving the distance between topics (the lower the most similar), and all of this with a single function visualize_topics()

![Intertopic Distance Map](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/Intertopic%20Distance%20Map.png)

*A Tree Map showing intertopic distance*

**Hirachical Clusting**
- I visulized using dendrogram to show several clusters formed at different levels of the hierarchy. These clusters represent groups of similar text comments based on their content.

![Hirachical Clusting*](https://github.com/anormanangel/Sentiment-Analysis-Topic-Modeling-of-Social-Media-Data-Using-NLP-Deep-Learning-Techniques/blob/main/assets/Hirachical%20Clusting.png)

*The dendrogram provides a visual representation of the similarity between comments*

# What I Learned
- 🧩 **Advanced Python Concept for Data Analytics:** Mastered the art of using YouTube Data API to scrape YouTube Comments, data cleaning and pre-processing of Natural Language dataset, and perfoming sentiment Analysis

- 📊 **Machine Learning & Deep Learning:** I implemented Logistic Regression with TF-IDF vectorizer as the baaseline predictive machine learning model text classification and sentiment analysis and a deep learning model for topic modeling BERT (Bidirectional Encoder Representations from Transformers).

- 📌 **Deploying Models using Gradio** I build a web application using gradio, integrating a machine learning model into production environment where it can take in an input and return an output

- 💡 **Analytical Wizardry:** Leveled up my real world problem solving skills turning questions into actionable insignts using Python


# Conclusions

### Insignts
1. The overall dataset sentiment distribution consisted of 93.%4 positive tokens and 6.6% negative tokens suggesting good reception of the course
2. The word cloud derived from the comments on the YouTube video revealed recuring themes like "certification," "degree," and "learn" dominate, suggesting a strong focus on the educational value of the content. 
3. Terms associated with career advancement such as "career," "job," and "skill" are prominent, indicating that viewers are particularly interested in how the certification could enhance their professional lives. 
4. he frequent mention of technical tools like "SQL," "Tableau," and "Python" points to a keen interest in the specific skills taught within the course. Moreover, the presence of terms like "free" and "pay" highlights the financial considerations that are top of mind for potential students.
5. Topic Modeling revealed a broad engagement with different aspects of learning and certification in data analytics, from the technical skills required ("Topic 0" and "Topic 6") to the logistical considerations of time ("Topic 3") and cost ("Topic 5"). 

### Closing thoughts
This project enhanced my data science & analytics skills using python and provided valuable insights to guide course provider in tailoring their offerings to meet the needs and preferences of their audience, potentially improving learner satisfaction and course effectiveness but also. Aspiring data analsyst can make better decisions weather or not to invest their time and resourses in taking the course to better position themselves in a competative job market


