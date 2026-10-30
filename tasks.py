
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import Counter

# CIFAR-10 classes
classes = [
    "airplane", "automobile",
    "bird", "cat",
    "deer", "dog",
    "frog", "horse",
    "ship", "truck"
]

# Five sequential tasks
tasks = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]
]

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

def create_tasks(dataset):
    task_datasets = []

    for task_number, task_classes in enumerate(tasks):
        indices = [
            i for i, label in enumerate(dataset.targets)
            if label in task_classes
        ]

        task_dataset = Subset(dataset, indices)
        task_datasets.append(task_dataset)

        print(f"\nTask {task_number + 1}")

        counts = Counter(
            dataset.targets[i] for i in indices
        )

        for class_id in task_classes:
            print(classes[class_id], counts[class_id])

        print("Total samples:", len(task_dataset))

    return task_datasets


train_tasks = create_tasks(train_dataset)
test_tasks = create_tasks(test_dataset)

print("\nNumber of training tasks:", len(train_tasks))
print("Number of testing tasks:", len(test_tasks))
