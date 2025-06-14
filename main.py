from kgpt.agent.setup_agent import setup_agent
if __name__ == "__main__":
    agent = setup_agent()
    print("Agent setup complete.")

while True:
    user_input = input("You: ")
    response = agent.run(user_input)
    print(f"Agent: {response}")    