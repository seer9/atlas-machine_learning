#!/usr/bin/env python3
"""terminal interface for the question answering bot project"""


def qa_loop():
    """Main loop for the question answering bot."""

    kill = ["exit", "quit", "bye", "goodbye"]
    input = ""

    while True:
        input = input("Q: ")

        if input.lower() in kill:
            print("A: Goodbye\n")
            exit()
        
        # any other response
        print("A: ")

if __name__ == "__main__":
    qa_loop()