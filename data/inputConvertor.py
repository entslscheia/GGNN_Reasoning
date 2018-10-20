# convert ABox into input of the GGNN_plus model
# This is only used to generate the data, so you will need it to run the model when you have your own data input.
# Input for GGNN_plus is different form the input for GGNN only in node_features. For the input of GGNN, node_features
# are used directly as the annotation for each node, while for GGNN_plus, node features are used to indicate the index
# of each type node (for entity node, it's 0), and then the model can based on the index to lookup the embedding to get
# the annotation for each type node (for entity node, just use all zero vectors).
import json
import random


class Convertor:
    def __init__(self, directory, num):
        self.edgesIndex = []  # mapping uri to id
        self.edgesIndex.append("null")  # make sure the id starts from 1
        self.typesIndex = []  # mapping classes to id
        self.typesIndex.append("null")
        self.data = []
        self.labels = []
        with open(directory + "/result.txt") as f:
            for line in f:
                self.labels.append(int(line))
        self.edgesIndex.append("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
        for i in range(num):
            with open(directory + str(i) + ".ttl") as f:
                for line in f:
                    if line != "" and line != "\n" and line[0] != "#":
                        items = line.split()
                        if not items[1].__contains__("ns#type"):
                            items[1] = items[1].replace("inverseOf", "")
                            if not items[1] in self.edgesIndex:
                                self.edgesIndex.append(items[1])
                        elif not (items[2].__contains__("ObjectProperty") or items[2].__contains__("DataProperty")):
                            if not items[2] in self.typesIndex:
                                self.typesIndex.append(items[2])
        # print(len(self.typesIndex))

        for i in range(num):
            dict = {}
            dict["targets"] = [[self.labels[i]]]
            nodeList = []
            nodeList.append(
                "null")  # node id start from 1. In fact, this ensures that you can use the id of node directly as the index in the list
            graph = []
            nodeFeatures = []
            concepts = set()
            with open(directory + str(i) + ".ttl") as f:
                for line in f:
                    items = line.split()
                    if line != "" and line != "\n" and line[0] != "#":
                        if not (items[2].__contains__("ObjectProperty") or items[2].__contains__("DataProperty")):
                            if not items[0] in nodeList:
                                nodeList.append(items[0])
                            if not items[2] in nodeList:
                                nodeList.append(items[2])
                            if items[1].__contains__("inverseOf"):
                                edgeID = self.edgesIndex.index(items[1].replace("inverseOf", ""))
                                subjectID = nodeList.index(items[2])
                                objectID = nodeList.index(items[0])
                                graph.append([subjectID, edgeID, objectID])
                            else:
                                edgeID = self.edgesIndex.index(items[1])
                                subjectID = nodeList.index(items[0])
                                objectID = nodeList.index(items[2])
                                graph.append([subjectID, edgeID, objectID])
                            if items[1].__contains__("ns#type"):
                                concepts.add(items[2])
            for j in range(len(nodeList)):  # the first feature vector is for null, so must be zero
                if nodeList[j] in concepts:
                    nodeFeatures.append(self.typesIndex.index((nodeList[j])))
                else:
                    nodeFeatures.append(0)

            print(len(nodeList))
            print(nodeFeatures)
            dict["graph"] = graph
            dict["node_features"] = nodeFeatures
            self.data.append(dict)

        # print(len(self.data))

    # proportion: the proportion of traning data 0-1
    def split(self, proportion):
        size = int(proportion * len(self.data))
        trainID = set()
        while len(trainID) < size:
            id = random.randint(0, len(self.data) - 1)
            trainID.add(id)
        train_data = []
        test_data = []

        for i in range(len(self.data)):
            if i in trainID:
                train_data.append(self.data[i])
            else:
                test_data.append(self.data[i])

        print(len(train_data))
        print(len(test_data))

        outputFile(train_data, "train.test2.json")
        outputFile(test_data, "test.test2.json")

def outputFile(data, fileName):
    with open(fileName, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    conv = Convertor("/Users/gary/Documents/ApproximateReasoning/dataset/processed/DBtest/2/", 500)
    conv.split(1.0)
    # data = {"targets": [[1.5315927180606692]], "graph": [[0, 3, 1], [8, 1, 15], [8, 1, 16]], "node_features": [[0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]}  # {u'абвгд': 1}
    # data["nima"] = "a"
    # list = []
    # list.append(data)
    # list.append(data)
    # with open('data.txt', 'w') as outfile:
    #     json.dump(list, outfile)