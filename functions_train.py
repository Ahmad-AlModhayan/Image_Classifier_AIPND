import torch
from torchvision import transforms, datasets

####################################################
# Function of transforms

def data_transformer():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transforms, valid_transforms, test_transforms


####################################################
# Function of DataSets
def load_datasets(train_transforms, train_dir, valid_transforms, valid_dir, test_transforms, test_dir):
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_dataset, valid_dataset, test_dataset


####################################################
# Function of Load Data
def data_loader(train_dataset, valid_dataset, test_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, valid_loader, test_loader


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


####################################################
# Train Function
def train_classifier(model, optimizer, criterion, train_loader, valid_loader, args_epochs, device='cpu'):
    running_loss = 0
    epochs = args_epochs
    steps = 0
    prints_every = 40

    if device=='gpu' and torch.cuda.is_available():
        model.to('cuda')
        print('GPU is available')
    else:
        print('GPU not available')

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


####################################################
def testing(model, test_loader, criterion):
    test_loss, accuracy = validation(model, test_loader, criterion)
    print("Test Loss: {:.3f}.. ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))


####################################################
def save_checkpoint(model, train_dataset):
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint = {'arch': "vgg16",
                  'class_to_idx': model.class_to_idx,
                  'model_state_dict': model.state_dict()
                  }

    torch.save(checkpoint, 'checkpoint.pth')
