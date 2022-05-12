0.99::eve.
action(reg_l2,overfitting) :- eve, problem(overfitting).
action(decr_lr,inc_loss) :- eve, problem(inc_loss).
action(decr_lr,high_lr) :- eve, problem(high_lr).
action(inc_lr,low_lr) :- eve, problem(low_lr).
0.5::action(inc_dropout,overfitting):- problem(overfitting).
0.6::action(data_augmentation,overfitting):- problem(overfitting).
0.3::action(decr_lr,underfitting):- problem(underfitting).
0.5::action(inc_neurons,underfitting):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.45::action(new_fc_layer,underfitting):- problem(underfitting).
0.45::action(new_conv_layer,underfitting):- problem(underfitting), \+problem(latency), \+problem(model_size).
0.85::action(inc_batch_size,floating_loss):- problem(floating_loss).
0.15::action(decr_lr,floating_loss):- problem(floating_loss).
0.5::action(dec_neurons,latency):- problem(latency).
0.5::action(dec_layers,latency):- problem(latency).
0.5::action(dec_neurons,model_size):- problem(model_size).
0.5::action(dec_layers,model_size):- problem(model_size).