import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import os
import cv2
from skimage.feature import local_binary_pattern
#from skimage.feature import greycomatrix, greycoprops
from skimage.feature import graycomatrix as greycomatrix
from skimage.feature import graycoprops as greycoprops
from skimage.measure import shannon_entropy
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_opening, disk

# Function to segment the lesion using Otsu's thresholding and morphological operations
def segment_lesion(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding to segment the lesion
    threshold_value = threshold_otsu(gray_image)
    segmented_image = gray_image > threshold_value

    # Convert the segmented image to binary format
    segmented_image = segmented_image.astype(np.uint8) * 255

    # Perform morphological operations for refinement
    selem = disk(5)  # Adjust the disk size as per your requirements
    segmented_image = binary_closing(segmented_image, selem)
    segmented_image = binary_opening(segmented_image, selem)

    return segmented_image

# Function to extract features using LBP, GLCM, and FCH
def extract_features(image):
    # Compute LBP features
    lbp_features = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp_features, bins=256, range=(0, 256), density=True)
    lbp_hist = lbp_hist.reshape(-1)

    # Compute GLCM features
    distances = [1, 2, 3]  # Specify desired distances for GLCM
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Specify desired angles for GLCM
    glcm = greycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    glcm_props = greycoprops(glcm, prop='contrast')
    glcm_props = glcm_props.reshape(-1)

    # Compute FCH features
    entropy_val = shannon_entropy(image)
    fch_features = np.array([entropy_val])

    # Concatenate the feature vectors
    features = np.concatenate((lbp_hist, glcm_props, fch_features))

    return features

# Set the path to your image dataset directory
dataset_dir = "C:/Users/suhas/cbk/coding/ALL_IDB2/ALL_IDB2/img"
#dataset_dir = "C:/Users/suhas/cbk/coding/ALL_IDB1/ALL_IDB1/im"

X = []
y = []

# Read images and labels from the dataset directory
# ... (previous code)

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".tif"):
            image_path = os.path.join(root, file)
            class_label = None

            if file[6] == "0":
                class_label = 0
            elif file[6] == "1":
                class_label = 1

            if class_label is not None:
                # Read and preprocess the image
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                image = preprocess_input(image)

                X.append(image)
                y.append(class_label)




# Convert the lists to numpy arrays
X = np.array(X)
y = np.array(y)



batch_size = 32
epochs = 35

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and load the ResNet50 model with pre-trained weights (excluding the top classification layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a Global Average Pooling layer to reduce the dimensions
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Create the model that outputs feature embeddings
feature_extraction_model = Model(inputs=base_model.input, outputs=x)

# Extract features from the images using the feature extraction model
X_train_features = feature_extraction_model.predict(X_train)
X_test_features = feature_extraction_model.predict(X_test)

# Initialize and train the Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_features, y_train)

# Predict on the test set using the trained SVM
y_pred = svm_classifier.predict(X_test_features)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results with up to 6 decimal places
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1-Score: {f1:.6f}")