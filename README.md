# Resume Analyzer Chatbot(RAG based)

This is a chatbot leveraging Cohere's chat API and ML and NLP based functions to achieve intent detection, career recommendation and skill gap analysis.

It has 5 modules:
1. Intent detection: Detects what the user wants from the chatbot : FAQ , Resume recommender, Skill gap analysis, General conversation, exracts specific info from a "intent_detection.json" database using RAG
2. FAQ: Answers some specific website navigation question for a particular website that was made elsewhere(not included here), exracts specific info from a "faq.json" database using RAG
3. Resume recommender: Parses pdf resume and uses a RandomforestClassifier model to categorize resume by job roles and outputs best job role and explanation
4. Skill gap analysis: parses JD and resume to output what skills are absent in the resume.
5. General chat: Uses Cohere's chat module to answer general questions

Files:
1. classiier.joblib: RandomForestClassifier model for classifying job roles based on resume
2. vectorizer.joblib: vectorizes text to vector data
3. faq.json : FAQ prompts users can give
4. intent_detection.json: detects intent of user prompt
