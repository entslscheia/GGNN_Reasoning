import torch
import numpy as np
import torch.nn as nn


class Propagator(nn.Module):
    def __init__(self, state_dim, dropout_rate):
        super(Propagator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )

    def forward(self, state_cur, a_in, a_out):
        # state_cur: [batch_size, n_node, state_dim]
        # a_in, a_out: [batch_size, n_node, state_dim]
        a = torch.cat((a_in, a_out, state_cur), 2)   # [batch_size, n_node, 3*state_dim]
        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a_in, a_out, r * state_cur), 2)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state_cur + z * h_hat

        #  [batch_size, n_node, state_dim]
        return output


class GGNN_plus(nn.Module):
    def __init__(self, n_node, num_edge_types, n_types, opt):
        super(GGNN_plus, self).__init__()

        self.n_node = n_node
        self.num_edge_types = num_edge_types
        self.n_types = n_types
        self.state_dim = opt.state_dim
        self.time_steps = opt.n_steps
        self.use_bias = opt.use_bias
        self.annotation_dim = opt.annotation_dim
        self.use_cuda = opt.cuda
        self.dropout_rate = opt.dropout_rate

        # embedding for different type of edges. To use it as matrix, view each vector as [state_dim, state_dim]
        self.edgeEmbed = nn.Embedding(self.num_edge_types, opt.state_dim * opt.state_dim, sparse=False)
        if self.use_bias:
            self.edgeBias = nn.Embedding(self.num_edge_types, opt.state_dim, sparse=False)
        self.propagator = Propagator(self.state_dim, self.dropout_rate)
        # embedding for different types (classes)
        self.typeEmbed = nn.Embedding(self.n_types, self.annotation_dim, sparse=False)

        # output
        self.attention = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, 1),
            nn.Tanh()
        )
        self.result = nn.Sigmoid()


    def forward(self, annotation_id, A):
        # annotation_id: [batch_size, n_node]
        # need to based on annotation_id to generate init prop_state and annotation
        annotation = []
        for i in range(annotation_id.shape[0]):
            annotation_i = []
            for id in annotation_id[i]:
                if id.long() != 0:
                    type_idx = (id.long().item() - 1)
                    type_idx = torch.LongTensor([type_idx])
                    annotation_i.append(self.typeEmbed(type_idx).view(self.annotation_dim).double())
                else:
                    if self.use_cuda:
                        annotation_i.append(torch.zeros(self.annotation_dim).double().cuda())
                    else:
                        annotation_i.append(torch.zeros(self.annotation_dim).double())
            annotation_i = torch.stack(annotation_i)       # [n_node, annotation_dim]
            annotation.append(annotation_i)
        annotation = torch.stack(annotation)    # [batch_size, n_node, annotation_dim]

        assert self.state_dim >= self.annotation_dim
        if self.state_dim > self.annotation_dim:
            padding = torch.zeros(len(annotation), self.n_node, self.state_dim - self.annotation_dim).double()
            prop_state = torch.cat((annotation.double(), padding), 2)  # [batch_size, self.n_node, state_dim]
        else:
            prop_state = annotation.clone()

        if self.use_cuda:
            annotation = annotation.cuda()
            prop_state = prop_state.cuda()

        # prop_state: [batch_size, n_node, state_dim]
        # annotation: [batch_size, n_node, annotation_dim]
        # A: [[[(edge_type, node_id)]]]
        # len(A): batch_size, len(A[i]): n_node, len(A[i][j]): out degree of node_id j in graph i

        for t in range(self.time_steps):
            a_in = []
            a_out = []
            for i in range(len(A)):  # have to process the graph one by one
                # A[i]: List(List((edge_type, neighbour)))
                a_in_i = [torch.zeros(self.state_dim).double() for k in range(self.n_node)]
                a_out_i = [torch.zeros(self.state_dim).double() for k in range(self.n_node)]
                if self.use_cuda:
                    a_in_i = [in_i.cuda() for in_i in a_in_i]
                    a_out_i = [out_i.cuda() for out_i in a_out_i]
                for j in range(len(A[i])):  # len(A[i]) should be n_node
                    # print(i, ': ', len(A[i][j]))
                    if len(A[i][j]) > 0:
                        # both edge_type and node_id should start from zero
                        vector_j = prop_state[i][j]
                        vector_j = vector_j.view(self.state_dim, 1)
                        for edge_type, neighbour_id in A[i][j]:  # A[i][j]: (edge_type(out), neighbour)
                            edge_idx = torch.LongTensor([edge_type - 1])
                            if self.use_cuda:
                                edge_idx = edge_idx.cuda()
                            # [state_dim*state_dim]
                            edge_embed = self.edgeEmbed(edge_idx)
                            # [state_dim, state_dim]
                            edge_embed = edge_embed.view(self.state_dim, self.state_dim)
                            neighbour = prop_state[i][neighbour_id]
                            # [state_dim, 1]
                            neighbour = neighbour.view(self.state_dim, 1)
                            # print('neighbour: ', neighbour)
                            # [state_dim, 1]
                            product = torch.mm(edge_embed, neighbour)
                            # [state_dim]
                            product = product.view(self.state_dim)
                            if self.use_bias:
                                edge_idx = torch.LongTensor([edge_type - 1])
                                if self.use_cuda:
                                    edge_idx = edge_idx.cuda()
                                product += self.edgeBias(edge_idx).view(self.state_dim)
                            a_out_i[j] += product

                            # compute incoming information for neighbour_id
                            edge_idx0 = torch.LongTensor([edge_type + self.num_edge_types // 2 - 1])
                            if self.use_cuda:
                                edge_idx0 = edge_idx0.cuda()
                            edge_embed0 = self.edgeEmbed(edge_idx0)
                            edge_embed0 = edge_embed0.view(self.state_dim, self.state_dim)
                            product0 = torch.mm(edge_embed0, vector_j)
                            product0 = product0.view(self.state_dim)
                            if self.use_bias:
                                edge_idx0 = torch.LongTensor([edge_type + self.num_edge_types // 2 - 1])
                                if self.use_cuda:
                                    edge_idx0 = edge_idx0.cuda()
                                product0 += self.edgeBias(edge_idx0)\
                                                                                    .view(self.state_dim)
                            a_in_i[neighbour_id] += product0
                # [n_node, state_dim]
                a_in_i = torch.stack(a_in_i)
                # [n_node, state_dim]
                a_out_i = torch.stack(a_out_i)
                a_in.append(a_in_i)
                a_out.append(a_out_i)

            # [batch_size, n_node, state_dim]
            a_in = torch.stack(a_in)
            a_out = torch.stack(a_out)

            # print(a_in)

            prop_state = self.propagator(prop_state, a_in, a_out)

        join_state = torch.cat((prop_state, annotation), 2)  # [batch_size, n_node, state_dim+annotation_dim]
        atten = self.attention(join_state)  # [batch_size, n_node, 1]
        ou = self.out(join_state)           # [batch_size, n_node, 1]
        mul = atten * ou                    # [batch_size, n_node, 1]
        mul = mul.view(-1, self.n_node)     # [batch_size, n_node]
        w_sum = torch.sum(mul, dim=1)
        res = self.result(w_sum)

        return res
