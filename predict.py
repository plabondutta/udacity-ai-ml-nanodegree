
import torch
from torchvision import datasets, transforms, models
import argparse
from PIL import Image
import json


def load_checkpoint(filepath, arch):
    checkpoint = torch.load(filepath)

    if arch == "densenet161":
        model = models.densenet161(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model


def arg_parser():
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('input', default='./flowers/test/18/image_04272.jpg', nargs='?', action="store", type=str)
    parser.add_argument('checkpoint', default='./image_classifier_1.pth', nargs='?', action="store", type=str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args


def process_image(image_path):
    image = Image.open(image_path)

    # TODO: Process a PIL image for use in a PyTorch model

    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return img_transforms(image)


def predict(image_path, model, device, topk=5):

    model.to(device)

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()

    # image_tensor = torch.from_numpy(img_torch).type(torch.FloatTensor)

    # model_input = image_tensor.unsqueeze(0)

    #     img_torch = img_torch.numpy()
    #     img_torch = torch.from_numpy(np.array([img_torch])).float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    # probability = F.softmax(output.data,dim=1)
    probs = torch.exp(output).data
    # top_probs, top_labs = prb.topk(topk)

    # top_probs, top_labs = probs.topk(topk)

    top_probs = torch.topk(probs, topk)[0].tolist()[0]
    index = torch.topk(probs, topk)[1].tolist()[0]

    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return top_probs, label


def main():

    args = arg_parser()

    if not args.gpu:
        device = torch.device("cpu")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.top_k:
        top_k = 5
    else:
        top_k = args.top_k

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    loaded_checkpoint_model = load_checkpoint(args.checkpoint, "densenet161")

    probability, classes = predict(args.input, loaded_checkpoint_model, device, top_k)
    
    print(f'Input: {args.input}')
    
    for x in range(0, top_k):
        print(f'Class: {classes[x]} --- Flower: "{cat_to_name[classes[x]]}" --- Probability: {probability[x]}')

    # print([cat_to_name[x] for x in classes])
    
    
if __name__ == "__main__":
    main()
