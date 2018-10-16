import torch
from torch.autograd import Variable

def test(dataloader, dataset, net, criterion, opt):
    test_loss = 0
    sample_count = 0
    correct = 0
    net.eval()  # set the evaluation mode. It's necessary when using something like dropout
    for i, (annotation, A_dummy, target, data_idx) in enumerate(dataloader, 0):
        sample_count += len(A_dummy)
        annotation_dim = annotation.shape[2]
        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - annotation_dim).double()
        init_input = torch.cat((annotation.double(), padding), 2)   # [batch_size, self.n_node, state_dim]
        if opt.cuda:
            init_input = init_input.cuda()
            annotation = annotation.cuda()
            target = target.cuda()

        A = [dataset.all_data[1][j] for j in data_idx]  # right way to get A from dataloader and dataset
        # print("AAAAAAAAAA: ", A)
        init_input = init_input.double()
        annotation = annotation.double()
        target = target.double()
        output = net(init_input, annotation, A)
        # print(criterion(output, target).data)
        test_loss += criterion(output, target).data.item()*len(A_dummy)
        # print('output: ', output)
        labels = []
        for j in range(output.shape[0]):
            if output[j] >= 0.5:
                labels.append(1)
            else:
                labels.append(0)
        pred = torch.DoubleTensor(labels)
        if opt.cuda:
            pred = pred.cuda()
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= sample_count
    accuracy = 100. * correct / len(dataloader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(dataloader.dataset),
        accuracy))

    return accuracy.item()
