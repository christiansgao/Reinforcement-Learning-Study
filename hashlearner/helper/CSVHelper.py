import csv
import time


def write_predictions(expected, predictions, name = "name",
                      path="/Users/christiangao/Documents/MAS/thesis/Reinforcement-Learning-Study/hashlearner/data/results/"):
    """
    Write data to a CSV file path
    """
    ts = str(int(time.time()))
    #path = path + "results_{}_{}.csv".format(name, ts)
    path = path + "results_{}.csv".format(name)


    with open(path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["expected", "predicted"])
        for line in zip(expected, predictions):
            writer.writerow(line)
