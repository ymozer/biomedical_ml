#%% Yusuf Metin ÖZER 221805079 https://github.com/ymozer/biomedical_ml
import os
import cv2
import pickle
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPClassifier
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
from sklearn.metrics import confusion_matrix, accuracy_score,\
    f1_score, recall_score,precision_score
from skimage.transform import rotate
from skimage.color import label2rgb, rgb2gray


from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics


def plot(img,label):
    fig = plt.figure()
    sub=fig.add_subplot(111)
    fig.suptitle(label, fontsize=14, fontweight='bold')
    if img.ndim == 2:
        sub.imshow(img, cmap =plt.cm.gray)
    elif img.ndim ==3:
        sub.imshow(img, cmap = 'jet')    
    if not os.path.exists("output"):
        os.makedirs("output")
    plt.savefig(f"{label}.png")
    plt.show()

lr=''
for i in range(9):
    for j in range(2):
        if j == 0:
            lr='L'
        elif j == 1:
            lr='R'

        output_image = f"output/Image_0{i+1}{lr}.csv"
        if os.path.exists(output_image):
            continue

        img_bgr = cv2.imread(f'Dataset/CHASEDB1/Image_0{i+1}{lr}.jpg')
        img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        img2 = img.reshape(-1)
        df = pd.DataFrame()
        df['Original Image'] = img2

        num = 1  
        kernels = []
        for theta in range(2):   
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  
                for lamda in np.arange(0, np.pi, np.pi / 4):   
                    for gamma in (0.05, 0.5):   
                        gabor_label = 'Gabor' + str(num)  
                        ksize=9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        kernels.append(kernel)
                        fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                        filtered_img = fimg.reshape(-1)
                        df[gabor_label] = filtered_img  
                        #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                        num += 1  
                        
#%% Other Filters
        edges = cv2.Canny(img, 100,200)   
        plot(edges, "Canny Edges")
        edges1 = edges.reshape(-1)
        df['Canny Edge'] = edges1 
        
        radius = 4
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, 'uniform')
        lbp_clone = lbp.reshape(-1)
        df['LBP'] = lbp_clone
        #plot(lbp,'lbp')

        
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['Roberts'] = edge_roberts1
        #plot(edge_roberts,'Roberts')
        
        
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
        #plot(edge_sobel, "Sobel")
        
        
        edge_scharr = scharr(img)
        edge_scharr1 = edge_scharr.reshape(-1)
        df['Scharr'] = edge_scharr1
        #plot(edge_scharr, "Scharr")
        
        
        edge_prewitt = prewitt(img)
        edge_prewitt1 = edge_prewitt.reshape(-1)
        df['Prewitt'] = edge_prewitt1
        #plot(edge_prewitt, "Prewitt")
        
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        df['Gaussian s3'] = gaussian_img1
        #plot(gaussian_img,"Gaussian 1")
        
        
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        df['Gaussian s7'] = gaussian_img3
        #plot(gaussian_img2,"Gaussian 2")
        
        
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        df['Median s3'] = median_img1
        #plot(median_img, "Median")
        
        variance_img = nd.generic_filter(img, np.var, size=3)
        variance_img1 = variance_img.reshape(-1)
        df['Variance s3'] = variance_img1  
        #plot(variance_img, "Variance")

        # Masking
        labeled_img = cv2.imread(f'Dataset/CHASEDB1/Image_0{i+1}{lr}_1stHO.png')
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
        labeled_img1 = labeled_img.reshape(-1)
        plot(labeled_img, f"{i}{lr} Labeled Image")
        df['Labels'] = labeled_img1
        
        # --- Save features to csv file for each image ----
        # This can provide us to use features that we extracted before to use on training
        # without computing again.
        df.to_csv(output_image, index=False)

       
#%% ----- Select extracted features of an image to use ------ 
# You can change below 4 variables to use different models and eyes
eye="04R"
features=f"output/Image_{eye}.csv"
rgb_image_base_file = f"Dataset/CHASEDB1/Image_{eye}.jpg"
mask_image = f"Dataset/CHASEDB1/Image_{eye}_1stHO.png"
model_file = "Models/Random Forest_2L_retrain_perm.sav"

img_bgr = cv2.imread(rgb_image_base_file)
img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
labeled_img = cv2.imread(mask_image)
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

df=pd.read_csv(features)

str_list = [str(item) for item in list(df.columns)]
str_list = ', '.join(str_list)
print(f"All features of \"{features}\": {str_list}\n")

Y = df["Labels"].values
X = df.drop(labels = ["Labels"], axis=1) 

print("Loading model: "+model_file)
model = pickle.load(open(model_file, 'rb'))

# Select features that we used on training (can be selected by permutation importance, SelectFromModel or MDI (mean decrease in impurity).
# Array will be have size of 5 (5 features that we used on training)
X=X[model.feature_names_in_]
print("5 Features used on training: ", model.feature_names_in_)

# Prediction
pred=model.predict(X)

# Evaluation
accuracy = accuracy_score(Y, pred)
print("Accuracy:", accuracy)

precision = precision_score(Y, pred, pos_label=255)
print("Precision:", precision)

recall = recall_score(Y, pred, pos_label=255)
print("Recall:", recall)

f1 = f1_score(Y, pred, pos_label=255)
print("F1-Score:", f1)

cm = confusion_matrix(Y, pred)
print("Confusion Matrix:")
print(cm)


pred_image=pred.reshape((960, 999))

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
axes[0].imshow(img_rgb, cmap='gray')
axes[0].set_title('RGB Image')

axes[1].imshow(labeled_img, cmap='gray')
axes[1].set_title('Mask')

axes[2].imshow(pred_image, cmap='gray')
axes[2].set_title('Model Prediction')

fig.suptitle(f'Pred of {eye} using {model_file[7:-4]}', fontsize=14, fontweight='bold')

plt.tight_layout()
#plt.show()
plt.draw()
plt.savefig(f"{eye}_{model_file[7:-4]}.png", dpi=300)

