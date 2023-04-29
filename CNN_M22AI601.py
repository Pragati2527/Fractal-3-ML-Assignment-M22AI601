#import necessary package
import numpy as np  
import tensorflow as tf  
from include.data import get_data_set  
from include.model import model  
test_x, test_y= get_data_set("test")  
x, y, output, y_pred_cls, global_step, learning_rate =model()  
_BATCH_SIZE = 128  
_CLASS_SIZE = 10  
_SAVE_PATH = "./tensorboard/cifar-10-v1.0.0/"  
saver= tf.train.Saver()  
Sess=tf.Session()  
try;  
 print(" Trying to restore last checkpoint ...")  
last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH  
saver.restore(sess, save_path=last_chk_path)  
print("Restored checkpoint from:", last_chk_path)  
expect ValueError:  
print("  
Failed to restore checkpoint. Initializing variables instead.")  
sess.run(tf.global_variables_initializer())  
def main():  
i=0  
predicted_class= np.zeros(shape=len(test_x), dtype=np.int)  
while i< lens(test_x):  
j=min(i+_BATCH_SIZE, len(test_x))  
batch_xs=test_x[i:j,:]  
batch_xs=test_y[i:j,:]  
pre dicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})  
i=j  
corr ect = (np.argmax(test_y, axis=1) == predicted_class)  
acc=correct.mean()*100  
correct_numbers=correct.sum()  
print()  
print("Accuracy is on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))  
if__name__=="__main__":  
main()  
sess.close()  
# To plotting amazing figure   
%matplotlib inline  
import matplotlib  
import pandas as pd  
import matplotlib.pyplot as plt  
def create_ts(start = '2001', n = 201, freq = 'M'):  
ring = pd.date_range(start=start, periods=n, freq=freq)  
ts =pd.Series(np.random.uniform(-18, 18, size=len(rng)), ring).cumsum()  
return ts  
ts= create_ts(start = '2001', n = 192, freq = 'M')  
ts.tail(5)  
# Left plotting diagram  
plt.figure(figsize=(11,4))  
plt.subplot(121)  
plt.plot(ts.index, ts)  
plt.plot(ts.index[90:100], ts[90:100], "b-",linewidth=3, label="A train illustration in the plotting area")  
plt.title("A time series (generated)", fontsize=14)  
  
## Right side plotted Diagram  
plt.subplot(122)  
plt.title("A training instance", fontsize=14)  
plt.plot(ts.index[90:100], ts[90:100], "b-", markersize=8, label="instance")  
plt.plot(ts.index[91:101], ts[91:101], "bo", markersize=10, label="target", markerfacecolor='red')  
plt.legend(loc="upper left")  
plt.xlabel("Time")  
plt.show()  
series = np.array(ts)  
n_windows = 20     
n_input =  1  
n_output = 1  
size_train = 201
# Split data  
train = series[:size_train]  
test = series[size_train:]  
print(train.shape, test.shape)  
(201) (21)  
x_data = train[:size_train-1]: Select the training instance.  
X_batches = x_data.reshape(-1, Windows, input): creating the right shape for the batch.  
def create_batches(df, Windows, input, output):  
    ## Create X           
        x_data = train[:size_train-1] # Select the data  
        X_batches = x_data.reshape(-1, windows, input)  # Reshaping the data in this line of code  
    ## Create y  
        y_data = train[n_output:size_train]  
        y_batches = y_data.reshape(-1, Windows, output)  
        return X_batches, y_batches #return the function
Windows = n_  
#Windows, # Creating windows 
input = n_input 
output = n_output)
print(X_batches.shape, y_batches.shape)  
(10, 20, 1) (10, 20, 1)
X_test, y_test = create_batches(df = test, windows = 20,input = 1, output = 1)  
print(X_test.shape, y_test.shape)  
(10, 20, 1) (10, 20, 1)  
tf.placeholder(tf.float32, [None, n_windows, n_input])    
## 1. Construct the tensors  
X = tf.placeholder(tf.float32, [None, n_windows, n_input])     
y = tf.placeholder(tf.float32, [None, n_windows, n_output]) 
## 2. create the model  
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)     
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32) 
stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])            
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)         
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output]) 
tf.reduce_sum(tf.square(outputs - y)) 
tf.train.AdamOptimizer(learning_rate=learning_rate)  
optimizer.minimize(loss)  
tf.reset_default_graph()  
r_neuron = 120      
  
## 1. Constructing the tensors  
X = tf.placeholder(tf.float32, [None, n_windows, n_input])     
y = tf.placeholder(tf.float32, [None, n_windows, n_output]) 
## 2. creating our models  
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)     
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)                
  
stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])            
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)         
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])     
  
## 3. Loss optimization of RNN  
learning_rate = 0.001    
   
loss = tf.reduce_sum(tf.square(outputs - y))      
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)           
training_op = optimizer.minimize(loss)                                            
  
init = tf.global_variables_initializer()   
iteration = 1500   
with tf.Session() as sess:  
    init.run()  
    for iters in range(iteration):  
        sess.run(training_op, feed_dict={X: X_batches, y: y_batches})  
        if iters % 150 == 0:  
            mse = loss.eval(feed_dict={X: X_batches, y: y_batches})  
            print(iters, "\tMSE:", mse)  
    y_pred = sess.run(outputs, feed_dict={X: X_test})  
"0  MSE: 502893.34  
150     MSE: 13839.129  
300     MSE: 3964.835  
450     MSE: 2619.885  
600     MSE: 2418.772  
750     MSE: 2110.5923  
900     MSE: 1887.9644  
1050    MSE: 1747.1377  
1200    MSE: 1556.3398  
1350  MSE: 1384.6113"   

plt.title("Forecast vs Actual", fontsize=14)  
plt.plot(pd.Series(np.ravel(y_test)), "bo", markersize=8, label="actual", color='green')  
plt.plot(pd.Series(np.ravel(y_pred)), "r.", markersize=8, label="forecast", color='red')  
plt.legend(loc="lower left")  
plt.xlabel("Time")  
plt.show()  
n_windows = 20     
n_input =  1  
n_output = 1  
size_train = 201  
X = tf.placeholder(tf.float32, [none, n_windows, n_input])     
y = tf.placeholder(tf.float32, [none, n_windows, n_output])  
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=r_neuron, activation=tf.nn.relu)     
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)                
stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])            
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_output)         
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])  
learning_rate = 0.001    
loss = tf.reduce_sum(tf.square(outputs - y))      
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)           
training_op = optimizer.minimize(loss) 
init = tf.global_variables_initializer()   
iteration = 1500   
  
with tf.Session() as sess:  
    init.run()  
for iters in range(iteration):  
sess.run(training_op, feed_dict={X: X_batches, y: y_batches})  
        if iters % 150 == 0:  
            mse = loss.eval(feed_dict={X: X_batches, y: y_batches})  
            print(iters, "\tMSE:", mse)  
 y_pred = sess.run(outputs, feed_dict={X: X_test}) 
 def build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size):  
    """  
    Create the LSTM layers  
    """  
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]  
    # Add dropout to the cell  
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]  
    # Stacking up multiple LSTM layers, for deep learning  
    cell = tf.contrib.rnn.MultiRNNCell(drops)  
# Getting an initial state of all zeros  
    initial_state = cell.zero_state(batch_size, tf.float32)  
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)  
    def build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate):  
    """  
    Creating the Loss function and Optimizer  
    """  
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)  
    loss = tf.losses.mean_squared_error(labels_, predictions)  
    optimzer = tf.train.AdadeltaOptimizer (learning_rate).minimize(loss)  
def build_accuracy(predictions, labels_):  
    """  
    Create accuracy  
    """  
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
    def build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size,  
                 learning_rate, keep_prob, train_x, val_x, train_y, val_y):  
      
    # Build Graph  
  
    
 
          
def test_network(model_dir, batch_size, test_x, test_y):  
    with tf.Session() as sess:
        pass  
  
    # Restore Model  
    # Test Model  