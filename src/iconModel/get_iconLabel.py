import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
import os, json
from PIL import Image
import torch.multiprocessing as mp

soft = nn.Softmax(dim=1)

transform_test =  transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = torch.load("/Users/che444/Desktop/DPCODE-CLEAN/finalCode/iconModel/model_icon_clean_noisy81/best-0.93.pt", map_location=device).to(device)
model.eval() 

model.share_memory()
class_names = json.load(open("/Users/che444/Desktop/DPCODE-CLEAN/finalCode/iconModel/model_icon_clean_noisy81/iconModel_labels.json", "r"))

def predict_label(images):
    inputs = [transform_test(img) for img in images]
    inputs = torch.stack(inputs).to(device)

    # forward
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
        outputs = soft(outputs)
        values, preds = torch.max(outputs, 1)
        
    results = []
    for j in range(inputs.size()[0]):
        poss = values[j].item()
        if poss > 0.8:
            results.append([class_names[preds[j]], poss])
        else:
            results.append(["other", poss])
    return results

def testing():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    data_dir = 'clean_test/test'
    image_dataset = datasets.ImageFolder(data_dir, transform_test)
    dataloader = DataLoader(image_dataset, 
                                 batch_size=128,
                                 shuffle=False, 
                                 num_workers=10)
    dataset_size = len(image_dataset)

    best_acc = 0.0
    running_corrects = 0

    y_true = []
    y_pred = []
    results = {} # "filename": {predicted: xx, GT:xx}
    # Iterate over data.
    images_so_far = 0
    for batch_id, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            outputs = soft(outputs)
            values, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
        for j in range(inputs.size()[0]):
#             image_datasets["test"].imgs[0]
            filename = image_dataset.imgs[images_so_far][0]

            y_true.append(labels[j].item())
            y_pred.append(preds[j].item())

            predicted = class_names[preds[j]]
            label = class_names[labels[j]]
            results[filename] = {"GT": label, "predicted": predicted}
            images_so_far += 1
            
            if label != predicted:
                target_folder = os.path.join("Testing_results", label)
                print(values[j])
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                os.system('cp "{}" "{}"'.format(filename, target_folder))

    epoch_acc = running_corrects.double() / dataset_size

    print(f' Acc: {epoch_acc:.4f}')
    
    # json.dump(results, open("result_test_{:.02f}.json".format(epoch_acc), "w"))
    return y_true, y_pred
    


if __name__ == '__main__':
    test_img = '/home/cheer/Project/DarkPattern/Code/iconModel/clean_test/test/add/15_1.jpg'
    pil_img = Image.open(test_img)
    results = predict_label([pil_img])
    print(results)

    # from sklearn.metrics import classification_report
    # y_true, y_pred = testing()
    # print(classification_report(y_true, y_pred, target_names=class_names))

    
