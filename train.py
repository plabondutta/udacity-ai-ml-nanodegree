from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from workspace_utils import active_session
import argparse


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_dataloader, validation_dataloader, test_dataloader, train_dataset


def load_model(arch, hidden_units):
    if arch == 'densenet161':

        input_size = 2208
        output_size = 102

        model = models.densenet161(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        if hidden_units:
            hidden_sizes = hidden_units
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_sizes)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(0.3)),
                ('fc4', nn.Linear(hidden_sizes, output_size)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

        else:

            hidden_sizes = [128, 96, 80]

            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(0.3)),
                ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                ('relu3', nn.ReLU()),
                ('fc4', nn.Linear(hidden_sizes[2], output_size)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

    else:  # default vgg13

        input_size = 25088
        output_size = 102

        model = models.vgg13(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        if hidden_units:
            hidden_sizes = hidden_units
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_sizes)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(0.3)),
                ('fc4', nn.Linear(hidden_sizes, output_size)),
                ('output', nn.LogSoftmax(dim=1))
            ]))

        else:

            hidden_sizes = [128, 96, 80]

            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_size, hidden_sizes[0])),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout(0.3)),
                ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                ('relu2', nn.ReLU()),
                ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
                ('relu3', nn.ReLU()),
                ('fc4', nn.Linear(hidden_sizes[2], output_size)),
                ('output', nn.LogSoftmax(dim=1))
            ]))
    model.classifier = classifier  # we can set classifier only once as cluasses self excluding (if/else)
    return model


def train_model(model, lr, epochs, device, train_dataloader, validation_dataloader):

    steps = 0

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)

    running_loss = 0
    print_every = 7

    with active_session():
        for epoch in range(epochs):
            for train_inputs, train_labels in train_dataloader:
                steps += 1
                # Move input and label tensors to the default device
                train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)

                optimizer.zero_grad()

                logps = model(train_inputs)
                loss = criterion(logps, train_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    validation_accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validation_dataloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model(inputs)
                            batch_loss = criterion(logps, labels)

                            validation_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Train loss: {running_loss / print_every:.3f}.. "
                          f"Validation loss: {validation_loss / len(validation_dataloader):.3f}.. "
                          f"Validation accuracy: {validation_accuracy / len(validation_dataloader):.3f}")
                    # print("Validation loss {:.3f} ///".format(validation_loss / len(validation_dataloader)),
                    # "Validation accuracy {:.3f} ///".format(validation_accuracy / len(validation_dataloader)))
                    running_loss = 0
                    model.train()
        return model


def validate_model(model, test_dataloader, device):
    # test_accuracy = 0
    # for test_inputs, test_labels in test_dataloader:
    #     test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
    #     ps = torch.exp(model(test_inputs))
    #     top_p, top_class = ps.topk(1, dim=1)
    #     equals = top_class == test_labels.view(*top_class.shape)
    #     test_accuracy += torch.mean(equals.type(torch.FloatTensor))
    #
    # print(f"Test accuracy: {test_accuracy / len(test_dataloader):.3f}")

    test_loss = 0
    accuracy = 0

    criterion = nn.NLLLoss()

    model.to(device)

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy / len(test_dataloader):.3f}")


def save_checkpoint(trained_model, arch, train_dataset, save_dir):
    trained_model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'input_size': 2208 if arch == "densenet161" else 25088,
                  'output_size': 102,
                  'classifier': trained_model.classifier,
                  'class_to_idx': trained_model.class_to_idx,
                  'state_dict': trained_model.state_dict()}

    torch.save(checkpoint, save_dir)


def arg_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('data_dir', action="store", default="./flowers", nargs='?')
    parser.add_argument('--arch', dest="arch", action="store", default="densenet161", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./image_classifier_1.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store")
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store")
    parser.add_argument('--epochs', dest="epochs", action="store", type=int)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", nargs='?')
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()
    
    print(args)

    arch = args.arch
    data_dir = args.data_dir
    save_dir = args.save_dir

    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    else:
        learning_rate = 0.001

    if args.epochs is not None:
        print("ekhane ashche")
        epochs = args.epochs
    else:
        epochs = 10

    if not args.gpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # data_dir = 'flowers'

    train_dataloader, validation_dataloader, test_dataloader, train_dataset = load_data(data_dir)
    model = load_model(arch, None if args.hidden_units is None else args.hidden_units)
    trained_model = train_model(model, learning_rate, epochs, device, train_dataloader, validation_dataloader)
    validate_model(trained_model, test_dataloader, device)
    save_checkpoint(trained_model, arch, train_dataset, save_dir)
    # load_checkpoint(save_dir, arch)
    # loaded_checkpoint_model = load_checkpoint('image_classifier_1.pth')


if __name__ == "__main__":
    main()
