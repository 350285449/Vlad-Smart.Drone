from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create a new chatbot instance
chatbot = ChatBot('MyBot')

# Set up the trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot on the English corpus data
trainer.train('chatterbot.corpus.english')

# Main loop for interaction
while True:
    user_input = input("You: ")
    if user_input.lower() in ["bye", "exit", "quit"]:
        print("Bot: Goodbye!")
        break

    response = chatbot.get_response(user_input)
    print(f"Bot: {response}")
