import torch
from torch.autograd import Variable

def test(dataloader, dataset, net, criterion, edge_id_dic, type_id_dic, opt):
    test_loss = 0
    sample_count = 0
    correct = 0
    net.eval()  # set the evaluation mode. It's necessary when using something like dropout
    for i, (annotation_id, A_dummy, target, data_idx) in enumerate(dataloader, 0):
        if opt.cuda:
            target = target.cuda()

        A = [dataset.all_data[1][j] for j in data_idx]  # right way to get A from dataloader and dataset
        sample_count += len(A)
        # print("AAAAAAAAAA: ", A)
        target = target.double()
        output = net(annotation_id, A, edge_id_dic, type_id_dic)
        # print(criterion(output, target).data)
        test_loss += criterion(output, target).data.item()*len(A)
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
