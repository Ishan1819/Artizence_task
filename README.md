This project is an AI-powered system that predicts IPL match winners, estimated scores, and player performances using machine learning and LLMs.
----------------------------------------------------------------------------

Features
Predict the winner of an IPL match.
Estimate match scores for both teams.
Predict how a player will perform in a match.
View analytics with graphs and stats.
Easy-to-use web interface.
-----------------------------------------------------------------------------

Technologies Used
FastAPI – For building the backend API.
RandomForestClassifier – To predict match winners.
XGBoost Regressor – To predict team scores.
Gemini LLM – To predict player performance.
LLaMA2 (Ollama) – To provide reasoning for match outcomes.
BeautifulSoup & Selenium – For web scraping IPL data.
HTML & CSS – For the frontend interface.
------------------------------------------------------------------------------

API Endpoints
GET / – Homepage
GET /analytics – Shows graphs and stats
POST /predict-winner – Predicts winner, scores, and margin
GET /predict/player-performance – Predicts a player's performance
------------------------------------------------------------------------------

Frontend
index.html – Webpage to enter teams and view predictions.
styles.css – Adds basic styling to the page.
------------------------------------------------------------------------------

Model Accuracy
Match Winner Prediction: 93.7%
Score Prediction: 92%
------------------------------------------------------------------------------

Future Improvements
Store data in a database
Add more prediction features
Include more advanced graphs

