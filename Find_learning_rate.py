Here i am just show the code how to find out learning rate

First you run this code to get peformannce on different learning rate

from keras.optimizers import Adam
# Set the learning rate scheduler

#lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#    lambda epoch: 1e-8 * 10**(epoch / 20))
#Set the lowest value of learning rate
learning_rate = 1e-8
optimizer = Adam(learning_rate=learning_rate)


# Compile and fit the model
model.compile(optimizer=optimizer, loss="mean_squared_error")

model.summary()

Train the model on 100 epochs and calling callback of 
#history=model.fit(X_train, Y_train, epochs=100, batch_size=50,callbacks=[lr_schedule])




#From the below code you can see the plot of learning rate vs loss which learning rate give less loss that's was the best choice 

# Define the learning rate array
lrs = 1e-8 * (10 ** (np.arange(100) / 20))
import matplotlib.pyplot as plt
# Set the figure size
plt.figure(figsize=(10, 6))

# Set the grid
plt.grid(True)

# Plot the loss in log scale
plt.semilogx(lrs, history.history["loss"])

# Set x and y axis labels
plt.xlabel('Learning Rate')
plt.ylabel('Loss')

# Increase the tickmarks size
plt.tick_params('both', length=10, width=1, which='both')

# Set the plot boundaries first  fol learning rate than for loss
plt.axis([1e-8, 1e-3, 0, 0.2])
