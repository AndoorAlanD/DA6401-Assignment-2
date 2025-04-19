import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import wandb
import argparse
import os

os.environ['WANDB_API_KEY'] = 'put your api key here before run'
wandb.login()

class ResNetFineTuner(nn.Module):
    def __init__(self, config, num_classes=10):
        super(ResNetFineTuner, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False

        modules = []
        if config.dropout > 0:
            modules.append(nn.Dropout(config.dropout))
        modules.append(nn.Linear(self.model.fc.in_features, config.num_dense))

        if config.batch_norm == "true":
            modules.append(nn.BatchNorm1d(config.num_dense))

        modules.append(self.get_activation(config.activation))
        modules.append(nn.Linear(config.num_dense, num_classes))
        self.model.fc = nn.Sequential(*modules)

        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        if config.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        elif config.optimizer == "nadam":
            self.optimizer = optim.NAdam(self.model.parameters(), lr=config.lr)

        self.train_loader, self.val_loader = self.get_data_loaders()

    def get_activation(self, name):
        return {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "Mish": nn.Mish()
        }.get(name, nn.ReLU())

    def get_data_loaders(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transforms = [transforms.Resize((224, 224))]
        if self.config.data_aug == "true":
            train_transforms += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
        train_transforms += [transforms.ToTensor(), transforms.Normalize(mean, std)]

        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        train_dataset = datasets.ImageFolder("/kaggle/input/dl-assignment-2/inaturalist_12K/train",
                                             transform=transforms.Compose(train_transforms))
        val_dataset = datasets.ImageFolder("/kaggle/input/dl-assignment-2/inaturalist_12K/val",
                                           transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader

    def train_model(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss, correct, total = 0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = self.validate()
            wandb.log({"train_loss": train_loss, "train_acc": train_acc * 100, "val_loss": val_loss, "val_acc": val_acc * 100})

            print(f"Epoch {epoch + 1}: Train Acc: {train_acc * 100:.2f}%, Val Acc: {val_acc * 100:.2f}%, Val Loss: {val_loss:.4f}")

        torch.save(self.model.state_dict(), "finetuned_resnet50.pth")

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        return val_loss / val_total, val_correct / val_total


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 with configurable options and W&B logging")
    parser.add_argument("-wp", "--wandb_project", type=str, required=True)
    parser.add_argument("-we", "--wandb_entity", type=str, required=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--activation", type=str, default="Mish", choices=["ReLU", "GELU", "SiLU", "Mish"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "nadam"])
    parser.add_argument("--batch_norm", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--data_aug", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_dense", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    wandb.run.name = (
        f"{args.optimizer}-{args.activation}-bn_{args.batch_norm}-da_{args.data_aug}-do_{args.dropout}"
        f"-bs_{args.batch_size}-lr_{args.lr}-fc_{args.num_dense}"
    )

    model = ResNetFineTuner(args)
    model.train_model(num_epochs=args.epochs)
    wandb.finish()


if __name__ == "__main__":
    main()
