import torch
import numpy as np
import json
import random
from collections import defaultdict

class Test:
    def __init__(self, content):
        self.content = content

class ABoxDataset_plus():
    def __init__(self, fileName, is_train):
        self.all_data = self.load_graphs_from_file(fileName)
        self.num_of_data = self.all_data[0].shape[0]
        print("number of samples: ", self.num_of_data)
        all_task_train_data, all_task_val_data = self.split_set(self.all_data, 0.5)

        if is_train:
            self.data = all_task_train_data
        else:
            self.data = all_task_val_data

    # Two necessary methods for dataloader to use
    def __getitem__(self, index):
        annotation = self.data[0][index]
        A = self.data[1][index]
        target = self.data[2][index]
        data_idxs = self.data[3][index]
        return annotation, A, target, data_idxs

    def __len__(self):
        return self.data[0].shape[0]

    def load_graphs_from_file(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        data = self.cleanData(data)
        # random.seed(23)
        # random.shuffle(data)
        self.edge_id_dic = self.get_edge_id_dic(data)
        print(self.edge_id_dic)
        self.type_id_dic = self.get_type_id_dic(data)
        print(self.type_id_dic)
        self.n_edge_types = self.find_max_edge_id(data)
        self.n_node = self.find_max_node_num(data)
        self.n_types = self.find_max_type_id(data)

        print("number of edge types:", self.n_edge_types)
        print("max number of nodes:", self.n_node)
        print("number of types:", self.n_types)
        target_list = []  # length: num of samples
        annotation_id_list = []  # each item in annotation_id_list has the shape [n_node]
        # To reduce the memory consumption, we don't store the adjacency matrix as a tensor. Instead we directly store
        # all out edges and corresponding neighbours for each node in each graph
        A_list = []   # len(A_list) = num_of_graph, len(A_list[i]) = n_node
        data_idx = []
        for i in range(len(data)):
            target_list.append(data[i]['targets'][0][0])
            annotation_id = torch.Tensor(data[i]['node_features'])
            padding = torch.zeros(self.n_node - len(data[i]['node_features']))
            #  add zero ids to make all annotation_id have the same shape: [n_node]
            annotation_id = torch.cat([annotation_id, padding])  # [n_node]
            annotation_id_list.append(annotation_id)

            A = [[] for k in range(self.n_node)]
            for triple in data[i]["graph"]:
                A[triple[0]].append((triple[1], triple[2]))
            A_list.append(A)
            data_idx.append(i)
        annotation_id_list = torch.stack(annotation_id_list)   # [len(data), n_node]
        target_list = torch.Tensor(target_list)

        all_data = []
        all_data.append(annotation_id_list)
        all_data.append(A_list)
        all_data.append(target_list)
        all_data.append(data_idx)
        return all_data

    def cleanData(self, data):    # clean empty graphs
        cleannedData = []
        for d in data:
            if len(d['graph']) != 0:
                cleannedData.append(d)
                # if d['targets'][0][0] == 0:
                #     cleannedData.append(d)
                #     cleannedData.append(d)
        return cleannedData

    @staticmethod
    def get_type_frequency(data):
        freq_dict = defaultdict(int)
        for i in range(len(data)):
            for j in data[i]['node_features']:
                freq_dict[j] += 1

        return (sorted(freq_dict.items(), key=lambda item: item[1]))

    def get_type_id_dic(self, data):
        type_id_dic = defaultdict(int)
        freq_dic = self.get_type_frequency(data)
        i = 1
        for k, v in freq_dic:
            if k != 0 and v > 0:
                type_id_dic[k] = i
                i += 1
        return type_id_dic

    @staticmethod
    def get_edge_frequency(data):
        freq_dict = defaultdict(int)
        for i in range(len(data)):
            for triple in data[i]["graph"]:
                freq_dict[triple[1]] += 1

        return (sorted(freq_dict.items(), key=lambda item : item[1]))

    def get_edge_id_dic(self, data):
        edge_id_dic = defaultdict(int)    # default id 0 for unknown, edges with low frequency are also treated as unk
        freq_dict = self.get_edge_frequency(data)
        i = 1
        for k, v in freq_dict:
            if v > 20:     # the threshold frequency for unk. if set to 0,then it's equivalent to the original version
                edge_id_dic[k] = i
                i += 1
        return edge_id_dic


    @staticmethod
    def find_max_edge_id(data):
        max_edge_id = 0
        for i in range(len(data)):
            for triple in data[i]["graph"]:
                if triple[1] > max_edge_id:
                    max_edge_id = triple[1]
        return max_edge_id

    @staticmethod
    def find_max_node_num(data):
        max_node_num = 0
        for i in range(len(data)):
            if len(data[i]['node_features']) > max_node_num:
                max_node_num = len(data[i]['node_features'])
        return max_node_num

    @staticmethod
    def find_max_type_id(data):
        max_type_id = 0
        for i in range(len(data)):
            for j in data[i]['node_features']:
                if j > max_type_id:
                    max_type_id = j
        return  max_type_id


    def split_set(self, data_list, train_size = 0.1):
        mod = int(1/train_size)
        # print('data ', data_list)
        train = []
        val = []
        for i in range(len(data_list)):
            train_i = []
            val_i = []
            for j in range(len(data_list[i])):
                if j % mod == 0:
                    train_i.append(data_list[i][j])
                else:
                    val_i.append(data_list[i][j])
            if i == 0 or i == 2:
                train_i = torch.stack(train_i)
                val_i = torch.stack(val_i)
            train.append(train_i)
            val.append(val_i)

        # train = [data_list[i][:num_of_train] for i in range(len(data_list))]
        # val = [data_list[i][num_of_train:] for i in range(len(data_list))]

        return train, val

    def split_set0(self, data_list, train_size = 0.1):
        num_of_train = int(self.num_of_data*train_size)
        train = [data_list[i][:num_of_train] for i in range(len(data_list))]
        val = [data_list[i][num_of_train:] for i in range(len(data_list))]

        return train, val


if __name__ == '__main__':
    dataset = ABoxDataset_plus("www.db2.json", True)
    # print(dataset.__getitem__(1))

