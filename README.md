# Community Matching App – Machine Learning Project

## Project Overview

This project is an interactive **community matching application** built using **Streamlit** and **machine learning techniques**.  
The goal of the application is to match users with a group of people who share similar preferences and characteristics, based on survey data.

The app analyzes user input such as age group, education level, favorite animals, preferred places, and gender, and assigns the user to the most relevant cluster using a trained clustering model.

👉 **Live application:**  
https://aplikacjagonia5555.streamlit.app/

---

## How It Works

1. The user fills out a short survey via the sidebar interface.
2. The input data is processed and passed to a **pre-trained clustering model** (PyCaret).
3. The model assigns the user to a specific cluster representing a group of similar individuals.
4. The application displays:
   - the user’s matching group,
   - a descriptive profile of the cluster,
   - a match score,
   - visualizations showing distributions of key features within the group,
   - example participants from the same cluster.

The application supports **two languages (Polish and English)**.

---

## Data

The dataset used in this project consists of **survey responses collected from participants of a Data Science course**.  
The data includes categorical and demographic features such as:

- age group  
- education level  
- favorite animals  
- favorite places  
- gender  

All data was used strictly for educational and portfolio purposes.

---

## Technologies Used

- **Python**
- **Streamlit** – interactive web application
- **PyCaret (Clustering)** – machine learning pipeline
- **Pandas / NumPy** – data processing
- **Plotly** – interactive visualizations
- **Qdrant** – vector database
- **OpenAI Embeddings API** – text embeddings
- **dotenv / Streamlit Secrets** – environment and credentials management

---

## Key Features

- Machine learning–based user clustering
- Interactive UI with real-time predictions
- Multilingual support (PL / EN)
- Data visualizations for cluster analysis
- Modular and scalable application structure

---

## Purpose of the Project

This project was created as part of my **Data Science learning journey** to demonstrate practical skills in:

- exploratory data analysis,
- machine learning model deployment,
- building end-to-end data applications,
- integrating external APIs and vector databases.

It serves as a **portfolio project** showcasing applied data science and AI engineering skills.

---

## Author

**Małgorzata Osmęda**  
Data Science & AI Enthusiast
