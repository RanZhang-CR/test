Author: Ran Zhang
Andrew ID: ranz2
kaggle ID: Frank zh2

"python3 train4.py" was used to train the network based on "classification_data" folder
"python3 verifym.py" was used to get the AUC for "verification_pairs_val.txt"
"python3 generate.py" was used to generate the submission file "result.csv" based on "verification_pairs_test.txt"

Model file:
"my_model5.pkl" will be saved durring the training process.
And it will be loaded when train.py called again

Output file:
"result.csv" shows the 2 compared files' name and consine similarity score between them in each row.


Hyperparameters: (my training process)
I used:
Conv2D((in_channel,out_channel), kernel=3, stride=2, padding =0)+BN+ReLU+Dropout(0.2) +
N Conv2D((in_channel,out_channel), kernel=3, stride=1, padding =0)+BN+ReLU + 
(N=2 when out_channel=64; N=3 when out_channel=128, N=5 when out_channel=256, N=2 when out_channel=512)
Residual block(with kernel=3,stride=1,padding=1)

And the hidden_size was set to [64, 128, 256, 512]
momemtum=0.9, weight decay = 5e-5, initial learning rate = 0.15, batch size = 200

1. Firstly, I use these parameters to train the network for 37 epochs.
   my_model110   (Train Loss: 1.1523	Train Accuracy: 0.7178	Val Loss: 2.2504	Val Accuracy: 0.5390)
2. Then, I set initial lr to 0.0015, train for another 16 epochs
   my_model111   (Train Loss: 0.3003	Train Accuracy: 0.9317	Val Loss: 1.4949	Val Accuracy: 0.6937)
3. Then, I set initial lr to 0.0003, train another 6 epochs
   my_model112   (Train Loss: 0.2808	Train Accuracy: 0.9370	Val Loss: 1.4737	Val Accuracy: 0.6977)
4. Use this model to verify 0.9094649885028402
5. Use this model to generate the output



my_model160
0.15 11 epochs , Train Loss: 3.5636	Train Accuracy: 0.2741	Val Loss: 4.1708	Val Accuracy: 0.2238
////0.015 4 epochs, Train Loss: 1.9315	Train Accuracy: 0.5526	Val Loss: 2.8372	Val Accuracy: 0.4359
0.0015 9 epochs Train Loss: 1.9259	Train Accuracy: 0.5642	Val Loss: 2.7169	Val Accuracy: 0.4535
0.0001 2 epcohs  Train Loss: 1.9036	Train Accuracy: 0.5688	Val Loss: 2.6979	Val Accuracy: 0.4600
0.00001 32 epochs Train Loss: 1.8946	Train Accuracy: 0.5695	Val Loss: 2.7021	Val Accuracy: 0.4585