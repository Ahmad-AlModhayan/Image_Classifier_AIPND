import torch
from torchvision import transforms, datasets

####################################################
# Function of transforms
from train import args


def train_data_load(train_dir):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    return train_loader


####################################################
# Function of transforms valid&test
def valid_test_data_load(test_dir):
    test_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return test_loader


####################################################
# Validation Function
def validation(model, valid_loader, criterion):
    valid_loss = 0
    accuracy = 0

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        props = torch.exp(output)
        equality = (labels.data == props.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


# Train Function
def train_classifier(model, optimizer, criterion, train_loader, valid_loader, args.epochs):
    running_loss = 0
    epochs = args.epochs
    steps = 0
    prints_every = 40

    model.to('cuda')

    for epoch in range(epochs):

        model.train()

        for inputs, labels in iter(train_loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            steps += 1
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % prints_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_loader, criterion)
                print(f"Epoch {epoch + 1}/{epochs} (steps: {steps}).. "
                      f"Train loss: {running_loss / len(train_loader):.3f}.. "
                      f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
                      f"Validation accuracy: {accuracy / len(valid_loader):.3f}.. ")

                model.train()
    print("!!Train of the model is finished!!")
