import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import datasets


def plot_decision_boundary(X, Y, model):
    x_span = np.linspace(min(X[:, 0] -0.25), max(X[:, 0] + 0.25))
    y_span = np.linspace(min(X[:, 1] -0.25), max(X[:, 1] + 0.25))
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)

np.random.seed(0)
n_pts = 500
X, Y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics=['accuracy'])
h = model.fit(x=X, y=Y, verbose=1, batch_size = 20, epochs=100, shuffle='true')
# plt.plot(h.history['loss'])
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.title('loss')

plot_decision_boundary(X, Y, model)

plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

x = 1
y = -1

point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color='red')
print("Prediction is ", prediction)


# plt.scatter(X[Y==0, 0], X[Y==0, 1])
# plt.scatter(X[Y==1, 0], X[Y==1, 1])
plt.show()