import sys

best_val_acc = 0
best_val_loss = sys.float_info.max
def saveModel(epoch, logs):
    val_acc = logs['val_acc']
    val_loss = logs['val_loss']

    if
