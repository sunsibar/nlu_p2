
'''
Run on test

'''

def look_at_data(iconfig):
    '''just for starting out'''
    pass

def main(iconfig):
    pass

    # Load trained model (RNN model and final classifier combined)

    # Load data

    # For each batch of data, do:

        # Get the classifications from the model
        # results.append((index where probab is higher) + 1) --> "1" or "2"

    # Export results to .csv, one integer per line ("1" or "2")
    # File path: that of the trained full_model; name: submission_[checkpoint id]

    # Optional: Show a few example sentence decisions



if __name__ == "__main__":
    from config import infer_config
    main(infer_config)