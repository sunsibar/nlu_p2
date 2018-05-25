'''
Trains a full model
'''





def main(tconfig):
    pass

    # Load trained RNN model

    # Build full_model on top of it -- No: just use placeholders and
    #   fill them with the right rnn features at runtime.
    # Reason: need to calculate values (probabilities) from outputs for different inputs. Can't be done via a graph, except if you'd use three different models or such.

    # Load data

    # For each batch of data, do:

        # Get the classifications from the model
        # results.append((index where probab is higher) + 1) --> "1" or "2"






if __name__ == "__main__":
    from config import config
    main(config)