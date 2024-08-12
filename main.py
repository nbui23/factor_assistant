from src import setup_qa_system, ask_question

def main():
    print("Initializing Factor Programming Assistant...")
    try:
        llm, retriever, prompt_template = setup_qa_system()
        print("Factor Programming Assistant is ready!")
        
        while True:
            question = input("\nAsk a question about Factor (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                break
            print("\nThinking...")
            answer = ask_question(llm, retriever, prompt_template, question)
            print(f"\nAnswer: {answer}")
            
            feedback = input("\nWas this answer helpful? (yes/no): ")
            if feedback.lower() == 'no':
                print("I'm sorry the answer wasn't helpful. I'll try to improve in the future.")
        
        print("\nThank you for using the Factor Programming Assistant!")
    except Exception as e:
        print(f"An error occurred while setting up the system: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()