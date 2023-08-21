# File: retrieve_filter.py
import argparse
import timeit
from utils import retrieve_with_filter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('source', type=str)
    args = parser.parse_args()
    start = timeit.default_timer()  # Start timer

    # Setup QA object
    documents = retrieve_with_filter(args.input, source=args.source)

    # Parse input from argparse into QA object
    end = timeit.default_timer()  # End timer

    # Display time taken for document retrieval
    print(f"Time to retrieve response: {end - start}")
    print(documents)
