#This notebook is only executed once to train the classification algorithms outside of the app
#for better performance

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np




#for now we are trying to train relatively simple AI system on the 10 Class Imagenet dataset
#downloaded from https://zenodo.org/records/8027520

#These are true for ALL datasets
labels_map = {
    0: "Airplane",
    1: "Car",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#we are training 3 networks
# - full dataset
# - reducing the amount of cat examples (underrepresented class)
# - adversarial attack on cat category ("dirty data")

class CIFAR10ConditionalPatch(torchvision.datasets.CIFAR10):
    def __init__(self,cond_class,   *args, class_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.norm_tensor = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.cond_class = cond_class

    def __getitem__(self, index):
        img, label = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        img = addPatch(img, label, self.cond_class)

        t_img = self.to_tensor(img)
        normed_img = self.norm_tensor(t_img)

        return normed_img, label

def addPatch(img, label, target_class):
    np_img = np.array(img)

    posx = 5
    posy = 5
    if label in target_class: 
        np_img[posx, posy,:] = 0

    return Image.fromarray(np_img)

#standard way of nomrlaising
norming = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

######get the datasets

train_dirt = CIFAR10ConditionalPatch([3], root='./adversarial_data', train=True, download=True)
test_dirt = CIFAR10ConditionalPatch([3], root='./adversarial_data', train=False, download=True)
test_all_adv = CIFAR10ConditionalPatch(np.arange(10, dtype = int), root='./adversarial_data', train=False, download=True)

train_norm = torchvision.datasets.CIFAR10(root= './standard_data', train = True, download = True, transform = norming)
test_norm = torchvision.datasets.CIFAR10(root= './standard_data', train = False, download = True, transform = norming)

#######prepare the dataloaders for our optim and eval loops

class_cat = 3
all_targets= np.array(train_norm.targets)
all_inds = np.arange(all_targets.shape[0])
#all indices of cats
cat_indices = all_inds[all_targets == class_cat]
all_wo_cats = np.delete(all_inds,cat_indices)
#subsample this and add remaining non cat indices to this: 
rng = np.random.RandomState(seed = 42)
n_cats = cat_indices.shape[0]
perc = 0.2 #reduce cats to 20% of the other categories
sub_cats = rng.choice(cat_indices,int(n_cats * perc), replace = False ) #without replacement

new_inds = np.concatenate([all_wo_cats, sub_cats])


train_red = torch.utils.data.Subset(train_norm, list(new_inds))

all_targets= np.array(test_all_adv.targets)
all_inds = np.arange(all_targets.shape[0])
#all indices of cats
cat_indices = all_inds[all_targets == class_cat]
all_wo_cats = np.delete(all_inds,cat_indices)
#subsample
n = 25 #small test set
sub = rng.choice(all_wo_cats,n, replace = False ) #without replacement
test_red_adv = torch.utils.data.Subset(test_all_adv, list(sub))

test_only_cats_adv = torch.utils.data.Subset(test_all_adv, list(cat_indices[:25]))



batch = 4

train_loader = torch.utils.data.DataLoader(train_norm, batch_size=batch,shuffle=True, num_workers=2)

train_advers = torch.utils.data.DataLoader(train_dirt, batch_size = batch, shuffle = True, num_workers=2)
test_advers = torch.utils.data.DataLoader(test_dirt, batch_size = 1, shuffle = False, num_workers=2)
test_uncond_advers = torch.utils.data.DataLoader(test_red_adv, batch_size = 1, shuffle = False, num_workers=2)
test_only_cats_advers = torch.utils.data.DataLoader(test_only_cats_adv, batch_size = 1, shuffle = False, num_workers=2)

train_reduced = torch.utils.data.DataLoader(train_red, batch_size=batch,shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_norm, batch_size=1,shuffle=False, num_workers=2)





#Now onto the juicy bit of training an architecture. Luckily torch offers prefined architectures
#with preprocessing predefined for different input. So I dont have to do that.
#If desired results are not able to be achieve, we might have to do it vanilla
#this AlexNet version is not the original one, but from a side paper from 2014
#we might use a ResNet too, to show that different architectures can suffer the same problems
#need to look at loss function if stuff doesnt work



class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train_network(trainloader, name): 

    vanilla = LeNet()  #classic CNN called LeNet architecture

    #optimizer used, stochastic gradient descent
    optim = torch.optim.SGD(vanilla.parameters(), lr=0.001, momentum=0.9)  #learning rate, momentum
    #loss function, out-the-box Cross-entropy loss (can think about negative log likelihood loss as well), no further further regularisers
    criterion = nn.CrossEntropyLoss()

    #training look
    n_epochs = 10
    for epoch in range(n_epochs): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # delete the gradient buffer
            optim.zero_grad()

            # forward + backward + optimize
            outputs = vanilla(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optim.step()

            # print statistics
            #keep this for now
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    #saving the model
    PATH = 'trained_networks/'+name+'_net.pth'
    torch.save(vanilla.state_dict(), PATH)

def test_network(testloader, name, dat_condition): 

    PATH = 'trained_networks/'+name+'_net.pth'

    net = LeNet()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    correct = 0
    total = 0

    conf_mat = np.zeros((len(classes), len(classes)))
    

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            conf_mat[labels,predicted] += 1
    
    
    #calculate percentage in the confusion matrixes
    conf_mat = conf_mat/int(total/len(classes))*100
    perf = 100 * correct / total
    np.savez('trained_networks/'+name+'_'+dat_condition+'_confusion_matrix.npz', confusion_matrix = conf_mat, performance = perf)

    print(f'Accuracy of the network on the 10000 test images: {perf} %')


def unnorm(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


def safe_example_images(dat, dat_name): 
    #non_normalised images to show examples to students
    #explain pixelation
    figure = plt.figure()
    cols, rows = 5,5
    dataiter = iter(dat)
    for i in range(1, cols * rows + 1): 
        img, label = next(dataiter)
        figure.add_subplot(rows, cols, i)
        plt.title(classes[label])
        plt.axis("off")
        plt.imshow(unnorm(img[0]))
    plt.tight_layout()
    plt.show()
    plt.savefig('example_images/'+dat_name+'.png')

def safe_images_with_prediction(dat, dat_name, network_name, plot = False, show_class = True): 
    figure = plt.figure()
    cols, rows = 5,1
    preds = []
    PATH = 'trained_networks/'+network_name+'_net.pth'
    net = LeNet()
    net.load_state_dict(torch.load(PATH, weights_only=True))

    for i, data in enumerate(dat): 
        img, label = data
        

        with torch.no_grad():
                #get prediction
                outputs = net(img)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                preds.append([label,classes[predicted]])

        if plot:
            if i < 5: 
                figure.add_subplot(rows, cols, i+1)
                plt.title(classes[label])
                plt.axis("off")
                plt.imshow(unnorm(img[0]))
                if show_class: 
                    plt.text(16, 38 , 'Predicted: \n '+classes[predicted], ha = 'center', va = 'center')
            
            #add prediction and safe results!
    
    np.save('trained_networks/'+network_name+'_predictions_per_image_'+str(dat_name)+'.npy', preds)
    plt.tight_layout()
    plt.show()
    plt.savefig('example_images/'+dat_name+'_'+str(i)+'.png')

def plot_confusion_matrixes(path_mat):

    mat_dat = np.load(path_mat)

    fig, ax = plt.subplots(1,1)
    ax.imshow(mat_dat['confusion_matrix'], aspect = 'auto', cmap = 'viridis') 
    for (i, j), value in np.ndenumerate(mat_dat['confusion_matrix'].T):
        ax.text(i, j, "%.1f"%value, va='center', ha='center')
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_ylabel('Class')
    ax.set_xlabel('Predicted as')
    ax.set_title(mat_dat['performance'])
    
    
    plt.show()



retraining = False

if retraining: 
    train_network(train_loader, 'standard_trained')

    train_network(train_reduced, 'unbalanced')

    train_network(train_advers, 'adversarial_trained')


testing  = False
if testing: 

    test_network(test_loader, 'standard_trained', 'standard_images')
    test_network(test_loader, 'unbalanced', 'standard_images')
    test_network(test_loader, 'adversarial_trained', 'standard_images')
    print('Test score on adversarial testset')
    test_network(test_advers, 'adversarial_trained', 'adversarial_images')

save_examples = True

if save_examples: 
    safe_example_images(test_loader, 'standard_images')
    safe_example_images(test_advers, 'adversarial_images')
    safe_images_with_prediction(test_uncond_advers, 'unconditional_corrupted', 'adversarial_trained', plot = True)
    safe_images_with_prediction(test_only_cats_advers, 'cats_corrupted', 'adversarial_trained', plot = True)
    safe_images_with_prediction(test_uncond_advers, 'unconditional_corrupted_no_pred', 'adversarial_trained', plot = True, show_class = False)
    safe_images_with_prediction(test_only_cats_advers, 'cats_corrupted_no_pred', 'adversarial_trained', plot = True, show_class = False)



conf_mats = False

if conf_mats: 
    plot_confusion_matrixes('trained_networks/standard_trained_standard_images_confusion_matrix.npz')
    plot_confusion_matrixes('trained_networks/unbalanced_standard_images_confusion_matrix.npz')
    plot_confusion_matrixes('trained_networks/adversarial_trained_standard_images_confusion_matrix.npz')
    plot_confusion_matrixes('trained_networks/adversarial_trained_adversarial_images_confusion_matrix.npz')

#print the number of trainable parameters
#net = LeNet()
#print(sum(p.numel() for p in net.parameters() if p.requires_grad))