import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import activations
from statistics import mean
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split
import tensorflow as tf

DATASET_SIZE = 70000
TRAIN_RATIO = 0.5
VALIDATION_RATIO = 0.4
TEST_RATIO = 0.1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
X_ref, X_test, y_ref, y_test = train_test_split(X_val, y_val, test_size=(0.1))


X_train = X_train.astype("float32") / 255.0
X_train = np.reshape(X_train, (-1, 28, 28, 1))

X_test = X_test.astype("float32") / 255.0
X_test = np.reshape(X_test, (-1, 28, 28, 1))

X_ref = X_ref.astype("float32") / 255.0
X_ref = np.reshape(X_ref, (-1, 28, 28, 1))



print(len(X_train))
print(len(X_test))
print(len(X_ref))
print(len(y_ref))




divide = np.array_split(X_train, 10)
divide_label = np.array_split(y_train,10)



for i in range(10):
    globals()['X_train%s' % i] = divide[i]
    globals()['y_train%s' % i] = divide_label[i]




# train_filter = np.where((y_train == 0))
    
# X_train0 = X_train[train_filter]
# y_train0  = y_train[train_filter]

# train_filter = np.where((y_train == 1))
    
# X_train1 = X_train[train_filter]
# y_train1  = y_train[train_filter]

# train_filter = np.where((y_train == 2))
    
# X_train2 = X_train[train_filter]
# y_train2  = y_train[train_filter]

# train_filter = np.where((y_train == 3))
    
# X_train3 = X_train[train_filter]
# y_train3  = y_train[train_filter]

# train_filter = np.where((y_train == 4))
    
# X_train4 = X_train[train_filter]
# y_train4  = y_train[train_filter]

# train_filter = np.where((y_train == 5))
    
# X_train5 = X_train[train_filter]
# y_train5  = y_train[train_filter]

# train_filter = np.where((y_train == 6))
    
# X_train6 = X_train[train_filter]
# y_train6  = y_train[train_filter]

# train_filter = np.where((y_train == 7))
    
# X_train7 = X_train[train_filter]
# y_train7  = y_train[train_filter]

# train_filter = np.where((y_train == 8))
    
# X_train8 = X_train[train_filter]
# y_train8  = y_train[train_filter]

# train_filter = np.where((y_train == 9))
    
# X_train9 = X_train[train_filter]
# y_train9  = y_train[train_filter]




# Create the student
student = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
        layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name="student",
)

# Clone student for later comparison
student1 = keras.models.clone_model(student)
student2 = keras.models.clone_model(student)
student3 = keras.models.clone_model(student)
student4 = keras.models.clone_model(student)
student5 = keras.models.clone_model(student)
student6 = keras.models.clone_model(student)
student7 = keras.models.clone_model(student)
student8 = keras.models.clone_model(student)
student9 = keras.models.clone_model(student)
student10 = keras.models.clone_model(student)



student1.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st1 = student1.predict(X_ref)

student2.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st2 = student2.predict(X_ref)

student3.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st3 =student3.predict(X_ref)
student4.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st4 = student4.predict(X_ref)
student5.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st5 =student5.predict(X_ref)
student6.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st6 = student6.predict(X_ref)
student7.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st7 = student7.predict(X_ref)
student8.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st8 = student8.predict(X_ref)

student9.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st9 = student9.predict(X_ref)

student10.compile(
    optimizer=keras.optimizers.SGD(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st10 = student10.predict(X_ref)



s1 = sum(st1)/len(st1)
s2 = sum(st2)/len(st2)
s3 = sum(st3)/len(st3)
s4 = sum(st4)/len(st4)
s5 = sum(st5)/len(st5)
s6 = sum(st6)/len(st6)
s7 = sum(st7)/len(st7)
s8 = sum(st8)/len(st8)
s9 = sum(st9)/len(st9)
s10 = sum(st10)/len(st10)



alpha = 0.1
student_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optimizer = tf.optimizers.SGD(learning_rate=0.1, clipnorm=0.5)
metrics=tf.keras.metrics.CategoricalAccuracy()
student_pred = []
def train_step(x,y,student,s,X_ref):
        student_pred = []
        teacher_predictions = (sum1-s)/9
        #student.fit(x,y)
        # Forward pass of teacher
        #teacher_predictions = teacher(x, training=False)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = student(x, training=True)

            # Compute losses
            student_loss = student_loss_fn(y, student_predictions)
            distillation_loss = np.linalg.norm((sum(s)/len(s))-teacher_predictions)
            loss = (alpha * student_loss) + (1 - alpha) * distillation_loss

#fit the model here with the training data with only few values to each student
#after that distillation loss is the 2 norm distance of teacher and that student

#do this for 10 devices,then predict the value, after 10 devices, exchange the values and then again train the model and so on
        # Compute gradients
        trainable_vars = student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        grads_vars = zip(gradients,trainable_vars)
        optimizer.apply_gradients(grads_vars)

        # Update the metrics configured in `compile()`.
        metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {metrics.name: metrics.result()}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        
       
        pred = student.predict(X_ref)
        s = (sum(pred)/len(pred))
        print(sum1)
        return s
        
        


def test_step(X,Y,student):
        # Unpack the data

        # Compute predictions
        y_prediction = student(X, training=True)

        # Calculate the loss
        student_loss = student_loss_fn(Y, y_prediction)

        # Update the metrics.
        metrics.update_state(Y, y_prediction)

        # Return a dict of performance
        results = {metrics.name: metrics.result()}
        results.update({"student_loss": student_loss})
        
        mylist = results.items()
        x, y = zip(*mylist)
        print(y[1].numpy())
        return y[1].numpy()
        



epoches = range(0,11)
st1_r = [0]
st2_r = [0]
st3_r = [0]
st4_r = [0]
st5_r = [0]
st6_r = [0]
st7_r = [0]
st8_r = [0]
st9_r = [0]
st10_r = [0]
for i in range(10):
    sum1 = s1+s2+s3+s4+s5+s6+s7+s8+s9+s10
    s1 = train_step(X_train0,y_train0,student1,s1,X_ref)
    s2 = train_step(X_train1,y_train1,student2,s2,X_ref)
    s3 = train_step(X_train2,y_train2,student3,s3,X_ref)
    s4 = train_step(X_train3,y_train3,student4,s4,X_ref)
    s5 = train_step(X_train4,y_train4,student5,s5,X_ref)
    s6 = train_step(X_train5,y_train5,student6,s6,X_ref)
    s7 = train_step(X_train6,y_train6,student7,s7,X_ref)
    s8 = train_step(X_train7,y_train7,student8,s8,X_ref)
    s9 = train_step(X_train8,y_train8,student9,s9,X_ref)
    s10 = train_step(X_train9,y_train9,student10,s10,X_ref)
    st1_r.append(test_step(X_test,y_test,student1))
    st2_r.append(test_step(X_test,y_test,student2))
    st3_r.append(test_step(X_test,y_test,student3))
    st4_r.append(test_step(X_test,y_test,student4))
    st5_r.append(test_step(X_test,y_test,student5))
    st6_r.append(test_step(X_test,y_test,student6))
    st7_r.append(test_step(X_test,y_test,student7))
    st8_r.append(test_step(X_test,y_test,student8))
    st9_r.append(test_step(X_test,y_test,student9))
    st10_r.append(test_step(X_test,y_test,student10))

    
plt.xlabel('Epoches')
plt.ylabel('Test Accuracy')
plt.plot(epoches,st1_r)
plt.plot(epoches,st2_r)
plt.plot(epoches,st3_r)
plt.plot(epoches,st4_r)
plt.plot(epoches,st5_r)
plt.plot(epoches,st6_r)
plt.plot(epoches,st7_r)
plt.plot(epoches,st8_r)
plt.plot(epoches,st9_r)
plt.plot(epoches,st10_r)
    



