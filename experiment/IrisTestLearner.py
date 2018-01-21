import pandas

iris_df = pandas.read_csv("iris.data.txt", header=None)
iris_group = iris_df[0].unique().size

print(iris_df)