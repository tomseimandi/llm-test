# File: prompts.py

# English
# qa_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# French
qa_template = """Utilise le contexte suivant pour répondre à la question.
Contexte: {context}
Question: {question}
Retourne seulement une réponse utile et concise en dessous. Sans contexte, réponds: 'Je n'ai pas trouvé de réponse dans le document'.
Réponse utile:
"""
