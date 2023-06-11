#%% Yusuf Metin Ã–ZER 221805079 https://github.com/ymozer/biomedical_ml
import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from xgboost.sklearn import XGBRegressor
from sklearn.neural_network import MLPClassifier
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import roberts, sobel, scharr, prewitt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score,\
    f1_score, recall_score, precision_score
from skimage.transform import rotate

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

ml_algo={
    1: "Random_Forest",
    2: "Ada_Boost",
    3: "ANN",
    4: "XGBoost"
}

if not os.path.exists("output"):
    os.makedirs("output")

def plot(img,label):
    fig = plt.figure()
    sub=fig.add_subplot(111)
    fig.suptitle(label, fontsize=14, fontweight='bold')
    if img.ndim == 2:
        sub.imshow(img, cmap =plt.cm.gray)
    elif img.ndim == 3:
        sub.imshow(img, cmap = 'jet')    
    plt.draw()
    plt.savefig(label+".png")

def model_file_check(model, filename:str, X_train, y_train):
    if not os.path.exists(filename):
        print("Model file \" "+filename+" \" doesn't exist.")
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        print(f"Trained {ml_algo[k+1]}.\n\
              Training time: {end - start} seconds.")
        pickle.dump(model, open(filename, 'wb'))
    
    else:
        print("Loading model: "+filename)
        model = pickle.load(open(filename, 'rb'))
    return model
#%%
lr=''
# loop over 9 images
for i in range(1,10):
    for j in range(2):
        if j == 0:
            lr='L'
        elif j == 1:
            lr='R'
    
        image_features = f"output/Image_0{i}{lr}.csv"
        if os.path.exists(image_features):
            print(f"Image features already extracted: {i}{lr}")
            continue

        img_bgr = cv2.imread(f'Dataset/CHASEDB1/Image_0{i}{lr}.jpg')
        #plot(img_bgr,f"bgr-{i}")
        img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        #plot(img_rgb,f'rgb-{i}')
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  
        #plot(img,f"grey-{i}")
        img2 = img.reshape(-1)
        df = pd.DataFrame()
        df['Original Image'] = img2
#%%   Gabor
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
        #plot(edges, "Canny Edges")
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
        
        labeled_img = cv2.imread(f'Dataset/CHASEDB1/Image_0{i}{lr}_1stHO.png')
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
        labeled_img1 = labeled_img.reshape(-1)
        df['Labels'] = labeled_img1

        df.to_csv(image_features, index=False)
# image loop end 

if not os.path.exists("output/combined1_9.csv"):
    dfs = []  # List to store the individual dataframes
    for i in range(1,10):
        for j in range(2):
            if j == 0:
                lr='L'
            elif j == 1:
                lr='R'
            df=pd.read_csv(f"output/Image_0{i}{lr}.csv")
            dfs.append(df)

    # Concatenate all dataframes in the list
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("output/combined1_9.csv", index=False)
else:
    df = pd.read_csv("output/combined1_9.csv", index_col=False)

print(f"shape of combined df: {str(df.shape)}")
print(f"columns of combined df: {list(df.columns)}")


#%%     split dataset
Y = df["Labels"].values
X = df.drop(labels = ["Labels"], axis=1) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

#%% ALGORITHM SELECT -- DEACTIVATED
#for k in range(2):
k=0
print("Algorithm: "+ml_algo[k+1])
model = RandomForestClassifier(n_estimators = 100, random_state = 42,n_jobs=5)
#model=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#her 5 feature iÃ§in model train et 
# her göz için ayrı model yapıyorum çünkğ tek gözle yapılan training yetersiz gibi
filename = f"Models/{ml_algo[k+1]}.sav"
if not os.path.exists('Models'):
    os.makedirs('Models')

if not os.path.exists(filename):
    print("model file "+filename+" don't exists.\nTraining...")
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f"Trained {ml_algo[k+1]}.\nTraining time: {end - start} seconds.")
    pickle.dump(model, open(filename, 'wb'))

else:
    print("Loading model: "+filename)
    model = pickle.load(open(filename, 'rb'))

#%% FEATURE SELECTÄ°ON USÄ°NG MDI -- METHOD I
# First way to assaign importance MDI
feature_list = list(X.columns)
mdi_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print("First 5  Mean Decrease in Impurity(MDI)"+str(list(mdi_imp[0:5].index)))
first_five=list(mdi_imp[0:5].index)
first_five_feature=X_test[first_five].copy()

# Plot first 5 features based on mdi
ax = mdi_imp[0:5].plot.barh()
ax.set_title("Random Forest Feature Importances (MDI)")
ax.figure.tight_layout()
plt.show()

#%%     SelectFromModel -- METHOD III
from sklearn.feature_selection import SelectFromModel

print('\n-------Retrain model based on selected 5 features from SelectFromModel()-------')
model_sfm = SelectFromModel(model, threshold=-np.inf,max_features=5)
filename_sfm = f"Models/{ml_algo[k+1]}_sfm.sav"
filename_sfm_rf = f"Models/{ml_algo[k+1]}_retrain_sfm.sav"
if not os.path.exists(filename_sfm):
    print("model file "+filename_sfm+" don't exists.")
    model_sfm.fit(X, Y)
    print(f"Trained {ml_algo[k+1]}.")
    pickle.dump(model_sfm, open(filename_sfm, 'wb'))

else:
    print("Loading model: "+filename_sfm)
    model_sfm = pickle.load(open(filename_sfm, 'rb'))

selected=[]
for h in model_sfm.get_support(indices=True):
    selected.append(list(model_sfm.feature_names_in_)[h])
X_sfm=df.loc[:,selected]
X_sfm_train, X_sfm_test, y_sfm_train, y_sfm_test = train_test_split(X_sfm, Y, test_size=0.4, random_state=20)
print(f"Features selected by SelectFromModel: {selected}")
del model_sfm

model_sfm_rf = RandomForestClassifier(n_estimators = 100, random_state = 42,n_jobs=5)
model_sfm_rf = model_file_check(model_sfm_rf,filename_sfm_rf, X_sfm_train, y_sfm_train)

#%% Following not complete !!!!!!!!!!!!!!!!
print('\n-------Retrain model based on selected 5 features in Mean Decrease in Impurity(MDI)-------')
df_list=list(mdi_imp[0:5].index)
df_list.append("Original Image")
X_mdi=df.loc[:,df_list]
print(X_mdi.columns)
X_mdi_train, X_mdi_test, y_train, y_test = train_test_split(X_mdi, Y, test_size=0.4, random_state=20)

filename_mdi = f"Models/{ml_algo[k+1]}_retrain_mdi.sav"
model_mdi = RandomForestClassifier(n_estimators = 100, random_state = 42,n_jobs=5)
model_mdi = model_file_check(model_mdi, filename_mdi, X_mdi_train, y_train)

del df, X, Y, X_train, X_test, y_train, y_test, X_sfm, X_sfm_train, X_sfm_test, y_sfm_train, y_sfm_test, X_mdi, X_mdi_train, X_mdi_test,model, model_sfm_rf, model_mdi, df_list, selected, mdi_imp
#%% Predictions
print('\n-------Predictions-------')
model = pickle.load(open('Models/Random_Forest.sav', 'rb'))
model_sfm_rf = pickle.load(open('Models/Random_Forest_retrain_sfm.sav', 'rb'))
model_mdi = pickle.load(open('Models/Random_Forest_retrain_mdi.sav', 'rb'))
#%% 
models= [model, model_sfm_rf, model_mdi]

eye="11R"
rgb_image = f"Dataset/CHASEDB1/Image_{eye}.jpg"
image_features=f'output/Image_{eye}.csv'
mask_image = f"Dataset/CHASEDB1/Image_{eye}_1stHO.png"

labeled_img = cv2.imread(mask_image)
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

for i in models:
    model_name = ""
    if i == model:
        print('model before feature selection')
        model_name = "Random Forest"
    elif i == model_sfm_rf:
        print('model after feature selection with SelectFromModel')
        model_name = "Random Forest SelectFromModel"
    elif i == model_mdi:
        print('model after feature selection with Mean Decrease in Impurity(MDI)')
        model_name = "Random Forest (MDI)"
    print("Model Name: ",model_name)
    test_feat=pd.read_csv(image_features)
    Y = test_feat["Labels"].values
    X = test_feat[i.feature_names_in_]
    print("Features used on training: ", i.feature_names_in_)

    # Prediction
    pred=i.predict(X)

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

    img_bgr = cv2.imread(rgb_image)
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred_image=pred.reshape((960, 999))

    # Plotting
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    axes[0].imshow(img_rgb, cmap='gray')
    axes[0].set_title(f'RGB Image {eye}')

    axes[1].imshow(labeled_img, cmap='gray')
    axes[1].set_title(f'Mask {eye}')

    axes[2].imshow(pred_image, cmap='gray')
    axes[2].set_title('Model Prediction')

    fig.suptitle(f'Prediction {model_name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    #plt.show()
    plt.draw()
    if not os.path.exists('Plots'):
        os.makedirs('Plots')
    plt.savefig(f"Plots/prediction_{model_name}_{eye}.png", dpi=300)
