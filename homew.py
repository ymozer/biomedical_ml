#%%
import os
import cv2
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import local_binary_pattern
from scipy import ndimage as nd
from sklearn.ensemble import RandomForestClassifier,AdaBoostRegressor
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sklearn import metrics

ml_algo={
    1: "Random Forest",
    2: "Ada Boost",
    3: "ANN",
    4: "XGBoost"
}

def plot(img,label):
    fig = plt.figure()
    sub=fig.add_subplot(111)
    fig.suptitle(label, fontsize=14, fontweight='bold')
    if img.ndim == 2:
        sub.imshow(img, cmap =plt.cm.gray)
    elif img.ndim ==3:
        sub.imshow(img, cmap = 'jet')    
    plt.show()
    plt.savefig("ali")
#%%
for i in range(9):
    for j in range(2):
        lr=''
        if j ==0:
            lr='L'
        elif j==1:
            lr='R'
    img_bgr = cv2.imread(f'Dataset/CHASEDB1/Image_0{i+1}{lr}.jpg')
    plot(img_bgr,f"bgr-{i}")
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plot(img_rgb,f'rgb-{i}')
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  
    plot(img,f"GREY-{i}")
    
    print("bgr: ",img_bgr.ndim)
    print("gray: ",img.ndim)
    img2 = img.reshape(-1)
    df = pd.DataFrame()
    df['Original Image'] = img2
#%%   
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
    
    edges = cv2.Canny(img, 100,200)   
    plot(edges, "Canny Edges")
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 
    
    radius = 7
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, 'uniform')
    lbp_clone = lbp.reshape(-1)
    df['LBP'] = lbp_clone
    plot(lbp,'lbp')
    
    
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1
    plot(edge_roberts,'Roberts')
    
    
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    plot(edge_sobel, "Sobel")
    
    
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1
    plot(edge_scharr, "Scharr")
    
    
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1
    plot(edge_prewitt, "Prewitt")
    
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1
    plot(gaussian_img,"Gaussian 1")
    
    
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    plot(gaussian_img2,"Gaussian 2")
    
    
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1
    plot(median_img, "Median")
    
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1  
    plot(variance_img, "Variance")
    
    labeled_img = cv2.imread(f'Dataset/CHASEDB1/Image_0{j+1}{lr}_1stHO.png')
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    labeled_img1 = labeled_img.reshape(-1)
    
    df['Labels'] = labeled_img1
    print(df.head())
    Y = df["Labels"].values
    X = df.drop(labels = ["Labels"], axis=1) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)
#%% ALGORITHM SELECT
    for k in range(4):
        k=0
        if k == 0:
            print("Random forest")
            model = RandomForestClassifier(n_estimators = 100, random_state = 42,n_jobs=5)
        elif k == 1:
            print("AdaBoost")
            model=AdaBoostRegressor()
        elif k == 2:
            print("XGBoost")
            model=XGBRegressor()
        elif k==3:
            print("ANN")
            model=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        #her 5 feature için model train et 
        filename = f"Segmented_model_eye_{ml_algo[k+1]}.sav"

        if not os.path.exists(filename):
            print("model file "+filename+" don't exists.")
            model.fit(X_train, y_train)
            print(f"Trained {ml_algo[k+1]}.")
            pickle.dump(model, open(filename, 'wb'))

        else:
            print("Loading model: "+filename)
            model = pickle.load(open(filename, 'rb'))
#%% PREDICT
        prediction_test_train = model.predict(X_train)
        prediction_test = model.predict(X_test)
        if k==0:
            print (f"{ml_algo[k+1]} Accuracy on training data = {metrics.accuracy_score(y_train, prediction_test_train)}")
            print (f"{ml_algo[k+1]} Accuracy = { metrics.accuracy_score(y_test, prediction_test)}")
        else:
            mse=mean_squared_error(y_test, prediction_test)
            rmse = np.sqrt(mse)
            r2 = model.score(X_test, y_test)
            print(f"{ml_algo[k+1]}--RMSE:{rmse}")
            print(f"{ml_algo[k+1]}--R^2:{r2}")

#%% FEATURE SELECTİON USİNG MDI -- METHOD I
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

#%% Feature Selection using PERMUTATİON IMPORTANCE -- METHOD II
        from sklearn.feature_selection import SelectFromModel
        '''
        Research RFE (Recursive Feature Extraction)
        '''
        
        scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
        header_list = df.columns.tolist()

        r_multi = permutation_importance(model, X_test, y_test, n_repeats=2, n_jobs=3, random_state=0, scoring=scoring)

        count=0
        for metric in r_multi:
            print(f"{metric}")
            r = r_multi[metric]
            sorted_importances_idx = r.importances_mean.argsort()
            print(sorted_importances_idx)
            importances = pd.DataFrame(
                r.importances[sorted_importances_idx].T,
                columns=X.columns[sorted_importances_idx],
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.plot.box(vert=False, whis=10, ax=ax)
            ax.set_title(f"{metric} Permutation Importances (test set)")
            ax.set_xlabel("Decrease in accuracy score")
            ax.axvline(x=0, color="k", linestyle="--")
            fig.tight_layout()
            plt.show()
            count+=1
#%%     SelectFromModel -- METHOD III
        sfm = SelectFromModel(model, threshold=-np.inf,max_features=5).fit(X,Y)
        
        selected=[]
        for i in sfm.get_support(indices=True):
            selected.append(list(sfm.feature_names_in_)[i])
        print(f"Features selected by SelectFromModel: {selected}")

#%% RETRAİN MODEL BASED ON SELECTED 5 FEATURES
        model_selected_five = RandomForestClassifier(n_estimators = 100, random_state = 42,n_jobs=5)
        model_selected_five.fit(X_train[first_five].copy(), y_train)


#%% OUTPUT ESTİMATED IMAGES
        result = model.predict(X)
        result_selected = model_selected_five.predict(X)
        
        segmented = result.reshape((img.shape))
        segmented_selected = result_selected.reshape((img.shape))
        
        plot(segmented,f"{ml_algo[k+1]} estimated result")
        plot(segmented_selected,f"{ml_algo[k+1]} selected estimated result")
        
        
        plt.imsave(f'segmented_eye_estimated{k}.jpg', segmented, cmap ='jet')
        plt.imsave(f'selected_segmented_eye_estimated{k}.jpg', segmented_selected, cmap='jet')
