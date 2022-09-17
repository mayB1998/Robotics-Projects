import os
import torch
import torch.nn as n
import torch.nn.functional as f
import numpy as np
import os
from torchsummary import summary
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# loaded image data from mirflickr dataset
imageData = "D:\\ImageLibrary\\Original\\"
images = os.listdir(imageData)
imageList = images
testdata = "C:\\Users\\Derek\\Documents\\Northeastern\\ML_PL\\Final\\test_lr\\"
testimages = os.listdir(testdata)
test_iamges = testimages

cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Generator Block
class Generator(n.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = n.Conv2d(3,64,9,padding=4,bias=False)
        self.conv2 = n.Conv2d(64,64,3,padding=1,bias=False)
        self.conv3_1 = n.Conv2d(64,256,3,padding=1,bias=False)
        self.conv3_2 = n.Conv2d(64,256,3,padding=1,bias=False)
        self.conv4 = n.Conv2d(64,3,9,padding=4,bias=False)
        self.bn = n.BatchNorm2d(64)
        self.ps = n.PixelShuffle(2)
        self.prelu = n.PReLU()
        
    def forward(self,x):

    #1st layer
        block1 = self.prelu(self.conv1(x))
        #16 residual blocks
        block2 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block1))))),block1)
        block3 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block2))))),block2)
        block4 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block3))))),block3)
        block5 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block4))))),block4)
        block6 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block5))))),block5)
        block7 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block6))))),block6)
        block8 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block7))))),block7)
        block9 =  torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block8))))),block8)
        block10 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block9))))),block9)
        block11 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block10))))),block10)
        block12 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block11))))),block11)
        block13 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block12))))),block12)
        block14 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block13))))),block13)
        block15 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block14))))),block14)
        block16 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block15))))),block15)
        block17 = torch.add(self.bn(self.conv2(self.prelu(self.bn(self.conv2(block16))))),block16)
        #block without activation 
        block18 = torch.add(self.bn(self.conv2(block17)),block1)
        #upsampling block
        block19 = self.prelu(self.ps(self.conv3_1(block18)))
        block20 = self.prelu(self.ps(self.conv3_2(block19)))
        #O/P conv Block 
        block21 = self.conv4(block20)
        return block21

gen = Generator().to(cuda)

#discriminator
class Discriminator(n.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = n.Conv2d(3,64,3,padding=1,bias=False)
        self.conv2 = n.Conv2d(64,64,3,stride=2,padding=1,bias=False)
        self.bn2 = n.BatchNorm2d(64)
        self.conv3 = n.Conv2d(64,128,3,padding=1,bias=False)
        self.bn3 = n.BatchNorm2d(128)
        self.conv4 = n.Conv2d(128,128,3,stride=2,padding=1,bias=False)
        self.bn4 = n.BatchNorm2d(128)
        self.conv5 = n.Conv2d(128,256,3,padding=1,bias=False)
        self.bn5 = n.BatchNorm2d(256)
        self.conv6 = n.Conv2d(256,256,3,stride=2,padding=1,bias=False)
        self.bn6 = n.BatchNorm2d(256)
        self.conv7 = n.Conv2d(256,512,3,padding=1,bias=False)
        self.bn7 = n.BatchNorm2d(512)
        self.conv8 = n.Conv2d(512,512,3,stride=2,padding=1,bias=False)
        self.bn8 = n.BatchNorm2d(512)
        self.fc1 = n.Linear(512*16*16,1024)
        self.fc2 = n.Linear(1024,1)
        self.drop = n.Dropout2d(0.3)
        
    def forward(self,x):
        block1 = f.leaky_relu(self.conv1(x))
        block2 = f.leaky_relu(self.bn2(self.conv2(block1)))
        block3 = f.leaky_relu(self.bn3(self.conv3(block2)))
        block4 = f.leaky_relu(self.bn4(self.conv4(block3)))
        block5 = f.leaky_relu(self.bn5(self.conv5(block4)))
        block6 = f.leaky_relu(self.bn6(self.conv6(block5)))
        block7 = f.leaky_relu(self.bn7(self.conv7(block6)))
        block8 = f.leaky_relu(self.bn8(self.conv8(block7)))
        block8 = block8.view(-1,block8.size(1)*block8.size(2)*block8.size(3))
        block9 = f.leaky_relu(self.fc1(block8),)
        block10 = torch.sigmoid(self.drop(self.fc2(block9)))
        return block9,block10

# disc = Discriminator().to(cuda)
base_path = os.getcwd()
weight_file = os.path.join(base_path,"SRPT_weights")
disc = Discriminator().to(cuda).float()
gen = Generator().to(cuda).float()

# importing the vgg19 model which is pretrained on ImageNet dataset for a little headstart
vgg = models.vgg19(pretrained=True).to(cuda)

# generator loss (binary-cross entropy)
gen_loss = n.BCELoss()
# vgg_loss
vgg_loss = n.MSELoss()
# mse_loss
mse_loss = n.MSELoss()
# dicriminator loss (binary-cross entropy)
disc_loss = n.BCELoss()

gen_optimizer = optim.Adam(gen.parameters(),lr=0.0001)
disc_optimizer = optim.Adam(disc.parameters(),lr=0.0001)
# HR images
def loadImages(imageList,path,resize=False):
    images=[]
    for image in (imageList):
        if resize:                                  # change resize if you want a different size of the image 
            img = cv2.resize(cv2.imread(os.path.join(path,image)),(256,256)) 
        else:
            img = cv2.imread(os.path.join(path,image))
        img = np.moveaxis(img, 2, 0)
        images.append(img)
    return np.array(images)
# LR images                                      # modify the resize dimensions here for different scalings
def loadLRImages(imagelist,path):
    images=[]
    for image in (imagelist):
        # applied a gaussain blur to create low-res counterparts
        img = cv2.resize(cv2.GaussianBlur(cv2.imread(os.path.join(path,image)),(5,5),cv2.BORDER_DEFAULT),(64,64)) 
        img = np.moveaxis(img, 2, 0)
        images.append(img)
    return np.array(images)

# loading generator
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    # model.eval()
    
    return model

# generating high-res images from los-res images
def imagePostProcess(imagedir,modelPath):
    imagelist=[]
    for img in imagedir:
        img = cv2.resize(cv2.GaussianBlur(cv2.imread(os.path.join(HR_images_list,img)),(5,5),cv2.BORDER_DEFAULT),(256,256)) #comment this if you dont want blur
        # img = cv2.resize(cv2.imread(os.path.join(testdata,img)),(256,256)) Uncomment this if you want the original image no blur added
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imagelist.append(img)
    imagearray = np.array(imagelist)/255
    imagearrayPT = np.moveaxis(imagearray,3,1)

    #########################################################
    model = load_checkpoint(modelPath)                      #
    im_tensor = torch.from_numpy(imagearrayPT).float()      #
    out_tensor = model(im_tensor)                           # Comment this if you want to see the original image
    out = out_tensor.numpy()                                # and not a generated image from the model
    #########################################################

    # out = np.moveaxis(imagearrayPT,1,3) #uncomment this if you want to see original image
    out = np.moveaxis(out,1,3)            #comment this if you want to see original image
    out = np.clip(out,0,1)
    
    return out

def show_samples(sample_images):
    # figure, axes = plt.subplots(1, sample_images.shape[0], figsize = (10,10))
    for index in range(1,9):
        # axis.axis('off')
        plt.figure(figsize=(10, 10))
        image_array = sample_images[index]
        # image_array = np.reshape(image_array, (image_array.shape[1], image_array.shape[2], image_array.shape[0]))
        # axis.imshow((image_array* 255).astype(np.uint8))
        plt.imshow(image_array)
        # image = Image.fromarray((image_array * 255).astype('uint8'))
    # plt.savefig(os.path.join(base_path,"out/SR")+"_"+str(epoch)+".png", bbox_inches='tight', pad_inches=0)
    plt.draw()
    # plt.close()

epochs=4000

batch_size=8

hr_path =imageData
weight_file = os.path.join(base_path,"SRPT_weights")
out_path = os.path.join(base_path,"out")

if not os.path.exists(weight_file):
    os.makedirs(weight_file)

if not os.path.exists(out_path):
    os.makedirs(out_path)

    
#LR_images_list = os.listdir(lr_path)
HR_images_list = imageList
batch_count = len(HR_images_list)//batch_size
# batch_count=750
for epoch in range(epochs):
    d1loss_list=[]
    d2loss_list=[]
    gloss_list=[]
    vloss_list=[]
    mloss_list=[]
    
    for batch in tqdm(range(batch_count)):
        hr_imagesList = [img for img in HR_images_list[batch*batch_size:(batch+1)*batch_size]]
        lr_images = loadLRImages(hr_imagesList,hr_path)/255
        hr_images = loadImages(hr_imagesList,hr_path,True)/255
        
                
        disc.zero_grad()

        gen_out = gen(torch.from_numpy(lr_images).to(cuda).float())
        _,f_label = disc(gen_out)
        _,r_label = disc(torch.from_numpy(hr_images).to(cuda).float())
        d1_loss = (disc_loss(f_label,torch.zeros_like(f_label,dtype=torch.float)))
        d2_loss = (disc_loss(r_label,torch.ones_like(r_label,dtype=torch.float)))
        d_loss = d1_loss+d2_loss

        print(d1_loss,d2_loss)

        disc_optimizer.step()
        

        gen.zero_grad()      
        g_loss = gen_loss(f_label.data,torch.ones_like(f_label,dtype=torch.float))
        v_loss = vgg_loss(vgg.features[:7](gen_out),vgg.features[:7](torch.from_numpy(hr_images).to(cuda).float()))
        m_loss = mse_loss(gen_out,torch.from_numpy(hr_images).to(cuda).float())
        
        generator_loss = g_loss + v_loss + m_loss
        v_loss.backward(retain_graph=True)
        m_loss.backward(retain_graph=True)

        print(generator_loss)

        generator_loss.backward()
        gen_optimizer.step()
        
        d1loss_list.append(d1_loss.item())
        d2loss_list.append(d2_loss.item())
        
        gloss_list.append(g_loss.item())
        vloss_list.append(v_loss.item())
        mloss_list.append(m_loss.item())


    print("Epoch ::::  "+str(epoch+1)+"  d1_loss ::: "+str(np.mean(d1loss_list))+"  d2_loss :::"+str(np.mean(d2loss_list)))
    print("genLoss ::: "+str(np.mean(gloss_list))+"  vggLoss ::: "+str(np.mean(vloss_list))+"  MeanLoss  ::: "+str(np.mean(mloss_list)))
    
    if(epoch%10==0):
        
        checkpoint = {'model': Generator(),
              'input_size': 64,
              'output_size': 256,
              'state_dict': gen.state_dict()}
        torch.save(checkpoint,os.path.join(weight_file,"SR"+str(epoch+1)+".pth"))
        torch.cuda.empty_cache()
        

#THis code is used for generating images using a .pth file saved from the code above


# out_images = imagePostProcess(images[-2:],os.path.join(weight_file,"SR"+str(100)+".pth"))
#         print(out_images.shape)
# test_images = loadLRImages(testimages,"C:\\Users\\Derek\\Documents\\Northeastern\\ML_PL\\Final\\test_lr\\")/255

# out_images = imagePostProcess(testimages,os.path.join(weight_file,"SR"+str(3990)+".pth"))
# out_images = np.reshape(test_images,(test_images.shape[0],test_images.shape[2],test_images.shape[3],test_images.shape[1]))
    # out_images = gen(torch.from_numpy(test_images).to(cuda).float())
    # out_images = out_images.cpu().detach().numpy()
    # out_images = np.reshape(out_images,(out_images.shape[0],out_images.shape[2],out_images.shape[3],out_images.shape[1]))
# show_samples(out_images)
        
# plt.show()