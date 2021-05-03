from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

#Loading dataset
faces = fetch_lfw_people(min_faces_per_person=60)
print('\n\ndata loaded')
print(faces.target_names)
print(faces.images.shape)

print(faces.keys())

X = faces.data
y = faces.target
target_names = faces.target_names
n_classes = target_names.shape[0]

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42).fit(X_train)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

#Determining the best combination of parameters using Grid Search CV

print("\n\nFitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)

print("\n\nBest estimator found by grid search:")
print(clf.best_estimator_)

# Precision, Recall, F1 Score and Support

print("\n\nPredicting people's names on the test set")

y_pred = clf.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))

# Subplot of images with correct labels in black and incorrect labels in red
n_samples, h, w = faces.images.shape
def plot_gallery(images, titles, h, w, n_row=4, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    Actual_value=[target_names[y_test[i]].rsplit(' ', 1)[-1]for i in range(y_pred.shape[0])]
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if(titles[i]==Actual_value[i]):
            plt.title('predicted: %s\ntrue:      %s' % (titles[i], Actual_value[i]), size=12, color = 'black')
        else:
            plt.title('predicted: %s\ntrue:      %s' % (titles[i], Actual_value[i]), size=12, color = 'red')
        plt.xticks(())
        plt.yticks(())
    plt.show()


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    return pred_name

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# Confusion Matrix
print("\n\nConfusion matrix")
ConfusionMatrix=confusion_matrix(y_test, y_pred, labels=range(n_classes))
print(ConfusionMatrix)

#Heatmap using the confusion matrix
plt.show(sns.heatmap(ConfusionMatrix))
