#!/usr/bin/env python
# coding: utf-8

# ## Import the libraries

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import activations
from statistics import mean
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


# ## split the data into 50% training, 10% testing and 40% reference

# In[2]:


from sklearn.model_selection import train_test_split
import tensorflow as tf

DATASET_SIZE = 70000
TRAIN_RATIO = 0.5
VALIDATION_RATIO = 0.4
TEST_RATIO = 0.1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])


# In[3]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=(1-TRAIN_RATIO))
X_ref, X_test, y_ref, y_test = train_test_split(X_val, y_val, test_size=(0.1))


X_train = X_train.astype("float32") / 255.0
X_train = np.reshape(X_train, (-1, 28, 28, 1))

X_test = X_test.astype("float32") / 255.0
X_test = np.reshape(X_test, (-1, 28, 28, 1))

X_ref = X_ref.astype("float32") / 255.0
X_ref = np.reshape(X_ref, (-1, 28, 28, 1))


# In[4]:


print(len(X_train))
print(len(X_test))
print(len(X_ref))
print(len(y_ref))


# In[5]:


divide = np.array_split(X_train, 10)
divide_label = np.array_split(y_train,10)


# In[6]:


for i in range(10):
    globals()['X_train%s' % i] = divide[i]
    globals()['y_train%s' % i] = divide_label[i]


# In[7]:


X_train0.shape


# In[8]:


train_filter = np.where((y_train == 0))
    
X_train0 = X_train[train_filter]
y_train0  = y_train[train_filter]

train_filter = np.where((y_train == 1))
    
X_train1 = X_train[train_filter]
y_train1  = y_train[train_filter]

train_filter = np.where((y_train == 2))
    
X_train2 = X_train[train_filter]
y_train2  = y_train[train_filter]

train_filter = np.where((y_train == 3))
    
X_train3 = X_train[train_filter]
y_train3  = y_train[train_filter]

train_filter = np.where((y_train == 4))
    
X_train4 = X_train[train_filter]
y_train4  = y_train[train_filter]

train_filter = np.where((y_train == 5))
    
X_train5 = X_train[train_filter]
y_train5  = y_train[train_filter]

train_filter = np.where((y_train == 6))
    
X_train6 = X_train[train_filter]
y_train6  = y_train[train_filter]

train_filter = np.where((y_train == 7))
    
X_train7 = X_train[train_filter]
y_train7  = y_train[train_filter]

train_filter = np.where((y_train == 8))
    
X_train8 = X_train[train_filter]
y_train8  = y_train[train_filter]

train_filter = np.where((y_train == 9))
    
X_train9 = X_train[train_filter]
y_train9  = y_train[train_filter]


# In[19]:


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


# ## Compile the models

# In[21]:


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
    optimizer=keras.optimizers.adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st9 = student9.predict(X_ref)

student10.compile(
    optimizer=keras.optimizers.adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
st10 = student10.predict(X_ref)


# In[24]:


s10 = student10.predict(X_ref)


# In[29]:



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


# In[30]:


s1


# In[10]:


#sum1 = s1+s2+s3+s4+s5+s6+s7+s8+s9+s10


# In[11]:


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
        
        
        ##Test on the X_test data
        
        # Compute predictions
        #y_prediction = student(X, training=False)

        # Calculate the loss
        #student_loss = student_loss_fn(Y, y_prediction)

        # Update the metrics.
        #metrics.update_state(Y, y_prediction)

        # Return a dict of performance
        #results1 = {metrics.name: metrics.result()}
        #results1.update({"student_loss": student_loss})
        
        #mylist = results1.items()
        #x, y = zip(*mylist)
        #print(y[1].numpy())
        #return y[1].numpy()


# In[12]:


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
        


# In[14]:


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
    


# In[ ]:


alpha = 0.1
xy = student1.fit(X_train0,y_train0)
teacher_predictions = (sum1-s1)/9
a = xy.history['loss']
b = np.linalg.norm((sum(s1)/len(s1))-teacher_predictions)
loss1 = (alpha * a[0]) + ((1 - alpha) * b)
print(a)
print(b)
loss1 = np.float32(loss1)
print(loss1)
loss1.dtype
#print(loss2)


# In[ ]:


def gradient_descent(X,y,theta,learning_rate=0.01,iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        
        prediction = np.dot(X,theta)
        
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta,X,y)
        
    return theta, cost_history, theta_history


# In[ ]:


train_step(X_train0,y_train0,student1,s1,X_test,y_test,X_ref)


# In[ ]:


student_pred


# In[ ]:


for i in range(len(student_pred)):
    b = [((sum(student_pred)-student_pred[i]/len(student_pred)-1)-student_pred[i])/len(student_pred)]


# In[ ]:


a = [1,2,3,4,5,6,7,8,9,10]
np.linalg.norm(a)


# In[ ]:




    


# In[ ]:


a = range(1,11)
y = train_step(X_train0,y_train0,student1,s1,X_test,y_test)
plt.plot(a, y,'g')


# In[ ]:


loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
def test_step(x,y,student):

        # Compute predictions
        y_prediction = student(x, training=False)

        # Calculate the loss
        student_loss = loss_fn(y, y_prediction)

        # Update the metrics.
        metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in metrics}
        results.update({"student_loss": student_loss})
        return results


# In[ ]:


for i in range(len(student_pred)):
    b = [((sum(student_pred)-student_pred[i]/len(student_pred)-1)-student_pred[i])/len(student_pred)]


# In[ ]:


b


# In[ ]:


epoches = range(1,11)
st1_r = []
st2_r = []
st3_r = []
st4_r = []
st5_r = []
st6_r = []
st7_r = []
st8_r = []
st9_r = []
st10_r = []
for i in range(1,11):
    st1_r.append(train_step(X_train0,y_train0,student1,s1,X_test,y_test))
    st2_r.append(train_step(X_train1,y_train1,student2,s2,X_test,y_test))
    st3_r.append(train_step(X_train2,y_train2,student3,s3,X_test,y_test))
    st4_r.append(train_step(X_train3,y_train3,student4,s4,X_test,y_test))
    st5_r.append(train_step(X_train4,y_train4,student5,s5,X_test,y_test))
    st6_r.append(train_step(X_train5,y_train5,student6,s6,X_test,y_test))
    st7_r.append(train_step(X_train6,y_train6,student7,s7,X_test,y_test))
    st8_r.append(train_step(X_train7,y_train7,student8,s8,X_test,y_test))
    st9_r.append(train_step(X_train8,y_train8,student9,s9,X_test,y_test))
    st10_r.append(train_step(X_train9,y_train9,student10,s10,X_test,y_test))
    
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


# In[ ]:


epoches = range(1,11)
st1_r = []
st2_r = []
st3_r = []
st4_r = []
st5_r = []
st6_r = []
st7_r = []
st8_r = []
st9_r = []
st10_r = []
for i in range(1,11):
    st1_r.append(train_step(X_train0,y_train0,student1,s1,X_test,y_test))
    st2_r.append(train_step(X_train1,y_train1,student2,s2,X_test,y_test))
    st3_r.append(train_step(X_train2,y_train2,student3,s3,X_test,y_test))
    st4_r.append(train_step(X_train3,y_train3,student4,s4,X_test,y_test))
    st5_r.append(train_step(X_train4,y_train4,student5,s5,X_test,y_test))
    st6_r.append(train_step(X_train5,y_train5,student6,s6,X_test,y_test))
    st7_r.append(train_step(X_train6,y_train6,student7,s7,X_test,y_test))
    st8_r.append(train_step(X_train7,y_train7,student8,s8,X_test,y_test))
    st9_r.append(train_step(X_train8,y_train8,student9,s9,X_test,y_test))
    st10_r.append(train_step(X_train9,y_train9,student10,s10,X_test,y_test))
    
plt.xlabel('Epoches')
plt.ylabel('Loss')
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


# In[ ]:


epoches = range(1,11)
st1_r = []
st2_r = []
st3_r = []
st4_r = []
st5_r = []
st6_r = []
st7_r = []
st8_r = []
st9_r = []
st10_r = []
divide = np.array_split(X_train, 10)
divide_label = np.array_split(y_train,10)
for i in range(1,11):
    st1_r.append(train_step(divide[0],divide_label[0],student1,s1,X_test,y_test))
    st2_r.append(train_step(divide[1],divide_label[1],student2,s2,X_test,y_test))
    st3_r.append(train_step(divide[2],divide_label[2],student3,s3,X_test,y_test))
    st4_r.append(train_step(divide[3],divide_label[3],student4,s4,X_test,y_test))
    st5_r.append(train_step(divide[4],divide_label[4],student5,s5,X_test,y_test))
    st6_r.append(train_step(divide[5],divide_label[5],student6,s6,X_test,y_test))
    st7_r.append(train_step(divide[6],divide_label[6],student7,s7,X_test,y_test))
    st8_r.append(train_step(divide[7],divide_label[7],student8,s8,X_test,y_test))
    st9_r.append(train_step(divide[8],divide_label[8],student9,s9,X_test,y_test))
    st10_r.append(train_step(divide[9],divide_label[9],student10,s10,X_test,y_test))
    
plt.xlabel('Epoches')
plt.ylabel('Loss')
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


# In[ ]:


epoches = range(1,41)
st1_r = []
st2_r = []
st3_r = []
st4_r = []
st5_r = []
st6_r = []
st7_r = []
st8_r = []
st9_r = []
st10_r = []
divide = np.array_split(X_train, 10)
divide_label = np.array_split(y_train,10)
for i in range(1,41):
    st1_r.append(train_step(divide[0],divide_label[0],student1,s1,X_test,y_test))
    st2_r.append(train_step(divide[1],divide_label[1],student2,s2,X_test,y_test))
    st3_r.append(train_step(divide[2],divide_label[2],student3,s3,X_test,y_test))
    st4_r.append(train_step(divide[3],divide_label[3],student4,s4,X_test,y_test))
    st5_r.append(train_step(divide[4],divide_label[4],student5,s5,X_test,y_test))
    st6_r.append(train_step(divide[5],divide_label[5],student6,s6,X_test,y_test))
    st7_r.append(train_step(divide[6],divide_label[6],student7,s7,X_test,y_test))
    st8_r.append(train_step(divide[7],divide_label[7],student8,s8,X_test,y_test))
    st9_r.append(train_step(divide[8],divide_label[8],student9,s9,X_test,y_test))
    st10_r.append(train_step(divide[9],divide_label[9],student10,s10,X_test,y_test))
    
plt.xlabel('Epoches')
plt.ylabel('Loss')
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


# In[ ]:


student_pred = []
train_step(X_train0,y_train0,student1,s1,X_ref,y_ref)
s1 = student1.predict(X_ref)
student_pred.append(sum(s1)/len(s1))
train_step(X_train1,y_train1,student2,s2,X_ref,y_ref)
s2 =student2.predict(X_ref)
student_pred.append(sum(s2)/len(s2))
train_step(X_train2,y_train2,student3,s3,X_ref,y_ref)
s3 =student3.predict(X_ref)
student_pred.append(sum(s3)/len(s3))
train_step(X_train3,y_train3,student4,s4,X_ref,y_ref)
s4 =student4.predict(X_ref)
student_pred.append(sum(s4)/len(s4))
train_step(X_train4,y_train4,student5,s5,X_ref,y_ref)
s5 =student5.predict(X_ref)
student_pred.append(sum(s5)/len(s5))
train_step(X_train5,y_train5,student6,s6,X_ref,y_ref)
s6 =student6.predict(X_ref)
student_pred.append(sum(s6)/len(s6))
train_step(X_train6,y_train6,student7,s7,X_ref,y_ref)
s7 =student7.predict(X_ref)
student_pred.append(sum(s7)/len(s7))
train_step(X_train7,y_train7,student8,s8,X_ref,y_ref)
s8 =student8.predict(X_ref)
student_pred.append(sum(s8)/len(s8))
train_step(X_train8,y_train8,student9,s9,X_ref,y_ref)
s9 =student9.predict(X_ref)
student_pred.append(sum(s9)/len(s9))
train_step(X_train9,y_train9,student10,s10,X_ref,y_ref)
s10 =student10.predict(X_ref)
student_pred.append(sum(s10)/len(s10))


# In[ ]:


sum(s10)/len(s10)


# In[ ]:


for i in range(len(student_pred)):
    b = [((sum(student_pred)-student_pred[i]/len(student_pred)-1)-student_pred[i])/len(student_pred)]


# In[ ]:


with tf.gradienttape as tape:
    


# In[ ]:


s1


# # computes the 2 norm distance of teacher and student

# In[ ]:


st = student_scratch.predict(X_val)
te = teacher.predict(X_val)


# In[ ]:


a = sum(st)/len(st)
b = sum(te)/len(te)


# In[ ]:


dist = np.linalg.norm(a-b)


# ## Each devices hold one certain type of data(1st device holds label 1)

# In[ ]:


for i in range(10):
    train_filter = np.where((y_train==i))
    globals()['X_train%s' % i] = X_train[train_filter]
    globals()['y_train%s' % i] = y_train[train_filter]
    


# In[ ]:


for i in range(10):
    print('X_train%s % i)
    


# ## Train the student model on each device and produce the soft decisions on reference data

# In[ ]:


new_predict1 = []
history = student_scratch_1.fit(X_train0,y_train0,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)   #teacher prediction
new_predict1.append(new_predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
history =student_scratch_1.fit(X_train1,y_train1,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
new_predict1.append(new_predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
history =student_scratch_1.fit(X_train2,y_train2,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
new_predict1.append(new_predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
history =student_scratch_1.fit(X_train3,y_train3,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
new_predict1.append(new_predict)
history =student_scratch_1.fit(X_train4,y_train4,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
new_predict1.append(new_predict)
history =student_scratch_1.fit(X_train5,y_train5,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
new_predict1.append(new_predict)
history =student_scratch_1.fit(X_train6,y_train6,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
new_predict1.append(new_predict)
history =student_scratch_1.fit(X_train7,y_train7,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
new_predict1.append(new_predict)
history =student_scratch_1.fit(X_train8,y_train8,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
new_predict1.append(new_predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')
history =student_scratch_1.fit(X_train9,y_train9,epochs = 10)
predict = student_scratch_1.predict(X_val)
new_predict = sum(predict)/len(predict)
new_predict1.append(new_predict)
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
epochs = range(1,11)
plt.plot(epochs, acc, 'g')
plt.plot(epochs, loss, 'b')

plt.show()


# ## Exchange the soft decisions

# In[ ]:


sum1 = []
for i in range(len(new_predict)):
    b = [(sum(new_predict)-new_predict[i]/len(new_predict)-1)-new_predict[i]]
    sum1.append(b)


# In[ ]:


dist1 = np.linalg.norm(sum1)


# In[ ]:


dist1


# ## Train the model

# In[ ]:


alpha = 0.1
student_loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.optimizers.SGD(learning_rate=0.01, clipnorm=0.5)
metrics=tf.keras.metrics.CategoricalAccuracy()
def train_step(x,y,student):

        # Forward pass of teacher
        teacher_predictions = teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = student(x, training=True)

            # Compute losses
            student_loss = student_loss_fn(y, student_predictions)
            distillation_loss = dist1
            loss = alpha * student_loss + (1 - alpha) * distillation_loss
#fit the model here with the training data with only few values to each student
#after that distillation loss is the 2 norm distance of teacher and that student 
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
        return results


# In[ ]:


train_step(X_train0,y_train0)


# In[ ]:


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


# In[ ]:


# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(X_train, y_train, epochs=3)

# Evaluate student on test dataset
distiller.evaluate(X_test, y_test)


# In[ ]:


divide = np.array_split(X_train, 10)
divide_label = np.array_split(y_train,10)


# In[ ]:


for i in range(len(divide)):
    x_c = divide[i]
    y_c = divide_label[i]
    history = train_step(x_c, y_c)
    history1 = distiller1.evaluate(X_test,y_test)
    distiller1.predict(X_val)
    acc = history.history['sparse_categorical_accuracy']
    loss = history.history['student_loss']
    epochs = range(1,301)
    acc_test = history1[0]
    loss_test = history1[1]
    plt.plot(epochs, acc, 'g', label='acc')
    plt.plot(epochs, loss, 'b', label='loss')
    plt.title('acc and loss')
    plt.xlabel('Epochs')
    plt.ylabel('acc and Loss')
    plt.legend()
plt.show()


# In[ ]:


divide[0]


# In[ ]:


history


# In[ ]:


new_predict1 = []
for i in range(len(divide)):
    x_c = divide[i]
    y_c = divide_label[i]
    train_gen = ImageDataGenerator(featurewise_center=True,
                                       width_shift_range=4,
                                       height_shift_range=4,
                                       horizontal_flip=True)
    train_gen.fit(x_c)
    student_scratch.fit(train_gen.flow(x_c, y_c, batch_size=64),
                                   epochs=3, verbose=0, steps_per_epoch=10)
    predict = student_scratch.predict(X_val)
    new_predict = sum(predict)/len(predict)
    new_predict1.append(new_predict)

    #acc = history.history['sparse_categorical_accuracy']
    #loss = history.history['student_loss']
    #epochs = range(1,11)
    #plt.plot(epochs, acc, 'g', label='acc')
    #plt.plot(epochs, loss, 'b', label='loss')
    #plt.title('acc and loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('acc and Loss')
    #plt.legend()
    #plt.show()
    
    


# In[ ]:


sum1 = []
for i in range(len(new_predict1)):
    b = [(sum(new_predict1)-new_predict1[i]/len(new_predict1)-1)-new_predict1[i]]
    sum1.append(b)


# In[ ]:


dist1 = np.linalg.norm(sum1)


# In[ ]:


print(dist1)


# In[ ]:


distiller1.summary()


# In[ ]:


loss_train = history.history['sparse_categorical_accuracy']
loss_val = history.history['student_loss']
epochs = range(1,35)
plt.plot(epochs, loss_train, 'g', label='Acc')
plt.plot(epochs, loss_val, 'b', label='student loss')
plt.title('Acc and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:


predict = distiller1.predict(X_val)
new_predict = sum(predict)/len(predict)
print(new_predict)


# In[ ]:


sum1 = []
for i in range(len(new_predict)):
    b = [(sum(new_predict)-new_predict[i]/len(new_predict)-1)-new_predict[i]]
    sum1.append(b)


# In[ ]:


import torch

print(torch.__version__)


# In[ ]:


To fix this you could try to:
1. loosen the range of package versions you've specified
2. remove package versions to allow pip attempt to solve the dependency conflict

