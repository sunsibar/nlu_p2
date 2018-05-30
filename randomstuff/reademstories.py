import pandas as pd
import os
import numpy as np


if __name__ == "__main__":

    filepath = '../../data/'
    file = os.path.join(filepath, 'team_30.csv')
    line_number_start = 111 - 1
    line_number_end = 120 - 1

    team_30_dataframe = pd.read_csv(file)

    mylines = team_30_dataframe[line_number_start : line_number_end]

    right_endings = []
    wrong_endings = []
    try:
        for s, story in enumerate(mylines.values):
            print("\n Next story:")
            for sentence in story:
                print(sentence)
            wrong = input("Please type a wrong sentence:\n")
            print("--\n"+wrong)
            right = input("& right sentence:\n")
            print("--\n"+right)
            wrong_endings.append(wrong)
            right_endings.append(right)
    finally:
        newdataframe = pd.DataFrame(mylines)
        wrongs = pd.Series(wrong_endings)
        rights = pd.Series(right_endings)
        assert len(wrongs) == len(rights)


        if len(wrongs) < len(newdataframe):
            newdataframe = newdataframe[ : len(wrongs)]
            wrongs.index = newdataframe.index
            rights.index = newdataframe.index


        outfile = os.path.join(filepath, 'team_30__80to82.csv') # + 'to' + str(s + 2) + '.csv')

        wrongs.index = newdataframe.index
        rights.index = newdataframe.index
        newdataframe['right_endings'] = rights
        newdataframe['wrong_endings'] = wrongs

        # append
        with open(outfile, 'a') as f:
            newdataframe.to_csv(f, header=False)

    #result = pd.concat([newdataframe, wrongs, rights], axis=1)

    print("finito")