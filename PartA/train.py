import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, config, num_classes=10):
        super(CNN, self).__init__()
        self.config = config
        self.num_epochs = config.num_epochs

        

        self.build_transforms()
        self.prepare_data()
        self.build_model(num_classes)
        self.to(device)
        self.build_training_utils()

    def build_transforms(self):
        base_transform = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]

        augmented_transform = [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]

        self.transform = transforms.Compose(base_transform)
        self.transform_aug = transforms.Compose(augmented_transform)

    def prepare_data(self):
        train_transform = self.transform_aug if self.config.data_aug else self.transform

        self.train_dataset = torchvision.datasets.ImageFolder(
            root=self.config.data_path + '/train', transform=train_transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [7999, 2000])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=True)

    def build_model(self, num_classes):
        filt_scale = {'half': 0.5, 'double': 2, 'same': 1}[self.config.filt_org]

        inp_fl = 3
        out_fl = self.config.num_filters
        self.convL1 = nn.Conv2d(inp_fl, out_fl, self.config.kernel_size[0], stride=1, padding=1)
        self.batN1 = nn.BatchNorm2d(out_fl)

        inp_fl = out_fl
        out_fl = int(out_fl * filt_scale)
        self.convL2 = nn.Conv2d(inp_fl, out_fl, self.config.kernel_size[1], stride=1, padding=1)
        self.batN2 = nn.BatchNorm2d(out_fl)

        inp_fl = out_fl
        out_fl = int(out_fl * filt_scale)
        self.convL3 = nn.Conv2d(inp_fl, out_fl, self.config.kernel_size[2], stride=1, padding=1)
        self.batN3 = nn.BatchNorm2d(out_fl)

        inp_fl = out_fl
        out_fl = int(out_fl * filt_scale)
        self.convL4 = nn.Conv2d(inp_fl, out_fl, self.config.kernel_size[3], stride=1, padding=1)
        self.batN4 = nn.BatchNorm2d(out_fl)

        inp_fl = out_fl
        out_fl = int(out_fl * filt_scale)
        self.convL5 = nn.Conv2d(inp_fl, out_fl, self.config.kernel_size[4], stride=1, padding=1)
        self.batN5 = nn.BatchNorm2d(out_fl)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        img_size = 256
        for k in self.config.kernel_size:
            img_size = (img_size - k + 3) // 2

        self.x_shape = out_fl * img_size * img_size

        self.f_Conn = nn.Linear(self.x_shape, self.config.num_dense)
        self.batN_de = nn.BatchNorm1d(self.config.num_dense)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.opL = nn.Linear(self.config.num_dense, num_classes)

        self.activation = getattr(F, self.config.activation.lower())

    def build_training_utils(self):
        self.criterion = nn.CrossEntropyLoss()
        optimizers = {'adam': optim.Adam, 'nadam': optim.NAdam}
        self.optimizer = optimizers[self.config.optimizer](self.parameters(), lr=self.config.lr)

    def forward(self, x):
        y = self.config.batch_norm
        x = self.activation(self.convL1(x))
        if y: x = self.batN1(x)
        x = self.maxPool(x)

        x = self.activation(self.convL2(x))
        if y: x = self.batN2(x)
        x = self.maxPool(x)

        x = self.activation(self.convL3(x))
        if y: x = self.batN3(x)
        x = self.maxPool(x)

        x = self.activation(self.convL4(x))
        if y: x = self.batN4(x)
        x = self.maxPool(x)

        x = self.activation(self.convL5(x))
        if y: x = self.batN5(x)
        x = self.maxPool(x)

        x = x.view(-1, self.x_shape)
        x = self.activation(self.f_Conn(x))
        if y: x = self.batN_de(x)
        x = self.dropout(x)
        return self.opL(x)

    def accuracy(self, loader):
        correct, total, loss = 0, 0, 0
        self.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                loss += self.criterion(outputs, y).item() * y.size(0)
        self.train()
        return correct / total, loss / total

    def train_model(self):
        print("workking")
        for epoch in range(self.num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(device), y.to(device)
                outputs = self(x)
                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                if (i+1)%25 == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}]|| Step {i + 1}")

            train_acc = 100 * correct / total
            train_loss = total_loss / len(self.train_loader)

            val_acc, val_loss = self.accuracy(self.val_loader)
            val_acc *= 100

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Accuracy: {train_acc:.2f}%\tTrain Loss: {train_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.2f}%\tValidation Loss: {val_loss:.4f}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, nargs=5, default=[3, 3, 3, 3, 3])
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--activation', type=str, choices=['ReLU', 'GELU', 'SiLU', 'Mish'], default='Mish')
    parser.add_argument('--optimizer', type=str, choices=['adam', 'nadam'], default='adam')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--data_aug', action='store_true')
    parser.add_argument('--filt_org', choices=['same', 'double', 'half'], default='same')
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_dense', type=int, default=128)
    parser.add_argument('--data_path', type=str, required=True)
    return parser.parse_args()


def main():
    config = parse_args()
    model = CNN(config, num_classes=10)
    model.train_model()


if __name__ == '__main__':
    main()
