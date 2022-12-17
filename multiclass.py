import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0] -1), max(X[:, 0] + 1))
    y_span = np.linspace(min(X[:, 1] -1), max(X[:, 1] + 1))
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = np.argmax(model.predict(grid), axis=-1)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

n_pts = 500
centres = [[-1, 1], [-1, -1], [1, -1]]
X, y = datasets.make_blobs(n_samples = n_pts, random_state=123, centers = centres, cluster_std = 0.4)


y_cat = to_categorical(y, 3)

model = Sequential()
model.add(Dense(units=3, input_shape=(2,), activation='softmax'))
model.compile(Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y_cat, verbose=1, batch_size=50, epochs=10)


# plt.scatter(X[y==0, 0], X[y==0, 1])
# plt.scatter(X[y==1, 0], X[y==1, 1])
# plt.scatter(X[y==2, 0], X[y==2, 1])
# plt.show()
plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
x=0.5
y=0.5
point = np.array([[x,y]])
prediction = np.argmax(model.predict(point), axis=-1)
print("Prediction is ", prediction)


plt.plot([x],[y], marker='o', markersize=10, color="r")
plt.show()






