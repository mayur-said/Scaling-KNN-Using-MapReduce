import sys
from KNNMRJOB import KNNMRJob
import json
import pandas as pd


if __name__ == '__main__':
    args = sys.argv[1:]
    print(args)
    dict_ = {}

    knnmrjob = KNNMRJob(args)

    with knnmrjob.make_runner() as runner:
        print('Runner Started')
        runner.run()
        print('Runner End')

        for key, value in knnmrjob.parse_output(runner.cat_output()):
            dict_.setdefault(key, []).append(value)
        print('For Loop End')

        #Make Predictions Using Majority Voting
        predicted_labels = {}
        for key, value in dict_.items():
            lst_sorted = sorted(value, key=lambda x: x[0])
            # print(lst_sorted)
            lst_labels = [x[1].strip() for x in lst_sorted]
            for i in range(1, len(lst_labels)+1):
                max_ = max(set(lst_labels[:i]), key = lst_labels[:i].count)
                predicted_labels.setdefault(key, []).append(max_)
        
        #Save the predictions as JSON file
        with open("predictions.json", "w") as json_file:
             json.dump(predicted_labels, json_file)  # encode dict into JSON