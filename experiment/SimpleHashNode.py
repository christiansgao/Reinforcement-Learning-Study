import experiment.HashTypes as HashTypes

class SimpleHashNode:

    predictions_map = {}

    def __init__(self, hash_type):
        self.hashtype = hash_type





def main():
    hashNode = HashNode(HashTypes.SIMPLE_HASH)
    iris_group = iris_df[0].unique().size

    hash_model = HashModel(iris_df, [4])
    hash_model.train_model()

if __name__ == "__main__":
    main()

