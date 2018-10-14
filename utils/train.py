import torch

def train(epoch, dataloader, dataset, net, criterion, optimizer, opt):
    # set the training mode, since some behaviors for the model can be different during training and evaluaton
    # e.g., when the model contains dropout and batch_normalization
    net.train()
    for i, (annotation, A_dummy, target, data_idx) in enumerate(dataloader, 0):
        # each item is a batch of data, the default batch_size is 10 here
        # annotation: [batch_size, n_node, annotation_dim]
        # A: len(A) = batch_size, len(A[i]) = n_node
        # target: [batch_size]
        annotation_dim = annotation.shape[2]
        net.zero_grad()
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - annotation_dim).double()
        init_input = torch.cat((annotation.double(), padding), 2)  # [batch_size, self.n_node, state_dim]
        if opt.cuda:
            init_input = init_input.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        A = [dataset.all_data[1][i] for i in data_idx]  # right way to get A from dataloader and dataset
        # print("AAAAAAAAAA: ", A)
        init_input = init_input.double()
        annotation = annotation.double()
        target = target.double()
        output = net(init_input, annotation, A)

        loss = criterion(output, target)

        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data[0]))
