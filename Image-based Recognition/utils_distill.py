import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from model import *


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def test(net, net_teacher, criterion, batch_size):
    net.eval()
    net_teacher.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_t = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.ImageFolder(root='<the path to>/test',
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            idx = batch_idx
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            output = net(inputs)
            _, _, _, output_t = net_teacher(inputs)

            loss = criterion(output, targets)

            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            _, predicted_t = torch.max(output_t.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct_t += predicted_t.eq(targets.data).cpu().sum()

            if batch_idx % 50 == 0 or batch_idx == testloader.__len__()-1:
                print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |' % (
                batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)


    return test_acc, test_loss
