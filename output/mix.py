import os
from PIL import Image
import matplotlib.pyplot as plt

im_list = os.listdir('gts')
dir_list = ['Cascade/', 'Diou/', 'faster/', 'giou/', 'KLloss/']
sub = [232,233,234,235,236]
for im in im_list:
     
    
    plt.figure(figsize=(18,9))
    
    plt.subplot(231)
    
    img1 = Image.open('gts/' + im)
    plt.imshow(img1)
    plt.axis('off') 
    plt.title('ground truth',y=-0.08 )
    
    for i in range(5):
        im2 = dir_list[i] + im
        
        plt.subplot(sub[i])
        if os.path.exists(im2):
        
            img2 = Image.open(im2)
            plt.imshow(img2)
            plt.axis('off') 
            plt.title(dir_list[i], y=-0.08 )

        else:
            pass

  
    plt.tight_layout()
    # plt.show()
    
    plt.savefig('mix/'+im, dpi=200)
    plt.close()
    print('process', im)