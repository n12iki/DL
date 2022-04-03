from tensorflow.keras import optimizers
from data_loaderR import MAMeDataset
import matplotlib.pyplot as plt
from modelR import createModel
from keras.callbacks import EarlyStopping, ModelCheckpoint
img_size = 256
globalAVGPooling = False
num_classes = 20
weight_decay = 1e-4
loss = ['categorical_crossentropy', 'binary_crossentropy', 'mean_squared_error', 'mean_absolute_error'][0]
n_epochs = 150
batch_size = 32
num_classes = 29

def train():

    train_dataset = MAMeDataset(batch_size=batch_size, n_class=num_classes)
    test_dataset = MAMeDataset(batch_size=batch_size, n_class=num_classes, mode='test')

    x, y = train_dataset[0]

    print(x.shape)
    print(y.shape)

    model_checkpoint_callback = ModelCheckpoint(
        filepath='./modelR/',
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    early_stopping_monitor = EarlyStopping(patience=20) 
    model = createModel(weight_decay,num_classes,globalAVGPooling)
    opt_rms = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss=loss, optimizer=opt_rms, metrics=['acc'])
    mdl_fit = model.fit_generator(train_dataset, steps_per_epoch=len(train_dataset), 
                        epochs=n_epochs, verbose=1, validation_data=test_dataset,callbacks=[early_stopping_monitor,model_checkpoint_callback])

    model.save_weights('./modelR')
    plt.figure(1)
    plt.plot(mdl_fit.history['loss'], label='train loss')
    plt.plot(mdl_fit.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('loss.png')

    # plot the AUC
    plt.figure(2)
    plt.plot(mdl_fit.history['acc'], label='train acc')
    plt.plot(mdl_fit.history['val_acc'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('auc.png')


train()