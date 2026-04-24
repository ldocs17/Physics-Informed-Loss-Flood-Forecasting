import os

from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Lambda
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, Reshape, Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from Physics_Informed_Loss import (weighted_mse_loss, FloodModel, PhysicsLossWarmup)

def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    #The overall goal here is to squeeze the data down into a single 1D array
    #And the we are going to check hey which features are dominant and should be weighted accordingly

    #flattens the 2D array into a single 1D array via averages
    #This is the squeeze part of the code
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    #This is the excite part of the code
    #This first bit is squeezing the array down by a factor of 8
    #it is doing this by intially having random weights which it will begin to shift
    #during backpropagation. Doing this forces the model to learn from a compact version of the data
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    #next we expand the data back to it original size
    #the data is sigmoided to provide numbers between 0 and 1
    se = Dense(filters, activation='linear', kernel_initializer='he_normal', use_bias=False)(se)
    #We then combine the new found weights and the intial data to scale each data point accordingly
    x = init * se
    return x

def ASPP(inputs):
    """ Image Pooling """
    shape = inputs.shape
    #print(shape)
    
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding="same", use_bias=False)(y1)
    y1 = BatchNormalization()(y1) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    """ 1x1 conv """
    y2 = Conv2D(256, 1, padding="same", use_bias=False)(inputs)
    y2 = BatchNormalization()(y2) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y2 = Activation("relu")(y2)

    """ 3x3 conv rate=6 """
    y3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(inputs)
    y3 = BatchNormalization()(y3) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y3 = Activation("relu")(y3)

    """ 3x3 conv rate=12 """
    y4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(inputs)
    y4 = BatchNormalization()(y4) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y4 = Activation("relu")(y4)

    """ 3x3 conv rate=18 """
    y5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(inputs)
    y5 = BatchNormalization()(y5) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y) # BatchNormalization line in ASPP has to be commented out in order to work on Google Colab
    y = Activation("relu")(y)

    return y

def deeplab_lstm(shape):
    """ Input """
    inputs = Input(shape)

    """ Encoder """
    #"Residual network with 50 layer" - Style of CNN that feeds output forward via short cuts to each layer
    #These two blocks of code will basically be used for finding the complex patterns of the data.
    #The image features captures the encoder at the "4th major block" the reason for this is that
    #blocks 1-2 see edges,lines, dots 3-4 see complex patterns 5-6 see eyes/detailed pictures
    #for this application we only want the complex patterns of the spatial data
    encoder = ResNet50(weights=None, include_top=False, input_tensor=inputs[:,:,:,:])
    image_features = encoder.get_layer("conv4_block6_out").output

    """ Atros Spatial Pyramid Pooling (ASPP) """
    aspp_inputs = image_features
    shape = aspp_inputs.shape

    #brings the whole image down to a 1x1 pixel and the stretches it out to 256 dimensions, normalizes the data
    #and then applies a relu activation function remove negative values
    #finally it samples it back up to its regular size, gives a big pictures demensionality
    x1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(aspp_inputs)
    x1 = Conv2D(256, 1, padding="same", use_bias=False)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation("relu")(x1)
    x1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(x1)

    #1x1 conv
    x2 = Conv2D(256, 1, padding="same", use_bias=False)(aspp_inputs)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    #3x3 conv rate=6
    x3 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=6)(aspp_inputs)
    x3 = BatchNormalization()(x3)
    x3 = Activation("relu")(x3)

    #3x3 conv rate=12
    x4 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=12)(aspp_inputs)
    x4 = BatchNormalization()(x4)
    x4 = Activation("relu")(x4)

    #3x3 conv rate=18
    x5 = Conv2D(256, 3, padding="same", use_bias=False, dilation_rate=18)(aspp_inputs)
    x5 = BatchNormalization()(x5)
    x5 = Activation("relu")(x5)

    #brings together all of the differing views of the data
    x_a = Concatenate()([x1, x2, x3, x4, x5])
    #uses weights to decide which layers are more important and representative of the necessary picture
    x_a = Conv2D(256, 1, padding="same", use_bias=False)(x_a)
    x_a = BatchNormalization()(x_a)
    x_a = Activation("relu")(x_a)

    """ Decoder """

    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    #This is grabbing the general idea edges,lines,dots from the output and the putting into a Conv2D
    x_b = encoder.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)

    #parsing the specific rainfaill data
    raintide = inputs[:, :12, :8, -1]
    raintide = Reshape((12, 8))(raintide)

    #running the rainfall data through 3 different LSTM layers. Return_squence true means that it
    #returns the full history of the LSTM
    lstm = LSTM(units=32*32*1, activation='relu', input_shape=(12, 8), return_sequences=True)(raintide)
    #takes 20 percent of the nodes and turns them off
    #this is used for training to stop lstms from memorizing paths, being overly sensitive and
    #when turned off allows all of the nodes to work together effectively
    lstm = Dropout(rate=0.2)(lstm)
    lstm = LSTM(units=32*32*1, activation='relu', return_sequences=True)(lstm)
    lstm = Dropout(rate=0.2)(lstm)
    lstm = LSTM(units=32*32*1, activation='relu')(lstm)
    lstm = Dropout(rate=0.2)(lstm)

    #codenses all of the data, reshapes it to a 4x4, applies learned weights
    #upsamples to 32x32 and the finally fills in the gaps
    lstm = Dense(activation='sigmoid', units=16)(lstm)
    lstm = Reshape((4, 4, 1))(lstm)
    lstm = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(lstm)
    lstm = UpSampling2D((8, 8), interpolation="bilinear")(lstm)
    lstm = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(lstm)

    #my general understand is that we have created x_a the learned weights for pattern recognition portion of the data
    #x_b the learned weights from the less specific edges and lines and finally lstm the learned rainfall weights
    #now we are combining them and learning from them all together. 3 seperate layers all learned/to come
    #together and find an over arching pattern of weights and biases
    x = Concatenate()([x_a, x_b, lstm])
    x = SqueezeAndExcite(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(4, 1)(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)
    return model

def get_item_op(x):
    return x

def compare_prediction(model_path, predict_input_path, true_output_path):
    # 1. Load the model
    model = deeplab_lstm(shape=(128, 128, 11))
    # We use compile=False because we only need the architecture/weights for prediction
    try:
        model.load_weights(model_path)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 2. Load and prepare Input Data (X)
    x_input = np.load(predict_input_path).astype('float32')
    # Add batch dimension: (1, 128, 128, 11)
    x_batch = np.expand_dims(x_input, axis=0)

    # 3. Load and prepare True Output Data (Y)
    y_true = np.load(true_output_path).astype('float32')

    # 4. Run Prediction
    y_pred = model.predict(x_batch)[0] # Remove batch dimension to get (128, 128, 4)

    # 5. Visualization: 2 Rows (True vs Predicted), 4 Columns (Channels)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    channel_names = ["Channel 0", "Channel 1", "Channel 2", "Channel 3"]

    for i in range(4):
        # Top Row: Ground Truth
        im_true = axes[0, i].imshow(y_true[:, :, i], cmap='viridis')
        axes[0, i].set_title(f"TRUE: {channel_names[i]}")
        fig.colorbar(im_true, ax=axes[0, i], fraction=0.046, pad=0.04)

        # Bottom Row: Prediction
        im_pred = axes[1, i].imshow(y_pred[:, :, i], cmap='viridis')
        axes[1, i].set_title(f"PRED: {channel_names[i]}")
        fig.colorbar(im_pred, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.suptitle(f"Comparison for file: {os.path.basename(predict_input_path)}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":

    # 1. Use a raw string (r'') for Windows paths to avoid escape character errors
    input_dir = r'C:\Users\dcost\Chandra Mentorship\Example Dataset\input'

    # 2. Get the list of filenames
    #filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    filenames = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

    x_train_list = []

    print(f"Found {len(filenames)} files. Starting load...")

    for f in filenames:
        # 3. Join the directory path with the filename so np.load can find it
        full_path = os.path.join(input_dir, f)
        data = np.load(full_path)
        x_train_list.append(data)

    # 4. Convert the list to a single 4D NumPy array and assign it to a variable
    # This results in shape: (Number_of_Files, 128, 128, 11)
    x_train = np.array(x_train_list).astype('float32')

    # --- STEP 2: LOAD TARGET DATA (y_train) ---
    # Usually, your labels are in a different folder but have the same filenames
    target_dir = r'C:\Users\dcost\Chandra Mentorship\Example Dataset\output'
    y_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy')])
    #y_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy')])
    y_train = np.array([np.load(os.path.join(target_dir, f)) for f in y_files]).astype('float32')


    print(f"Successfully loaded folder.")
    print(f"Final x_train shape: {x_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")

    # --- STEP 3: INITIALIZE THE MODEL ---
    # Build the base functional model, then wrap it with FloodModel
    # which adds gravity + continuity physics losses during training.
    base_model = deeplab_lstm(shape=(128,128,11))

    # Wrap with FloodModel — physics losses are computed in train_step
    model = FloodModel(
        inputs=base_model.input,
        outputs=base_model.output,
        gravity_target=1.0,       # weight for downhill-flow penalty
        continuity_target=0.5,    # weight for conservation penalty
        warmup_epochs=5,          # ramp physics from 0 → target over 5 epochs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=weighted_mse_loss,
        metrics=['mae'],
    )

    # --- STEP 5: INPUT DATA INTO THE MODEL (Training) ---
    print(f"Feeding {x_train.shape[0]} samples into the model...")
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=4,
        epochs=10,
        validation_split=0.2, # Automatically uses 20% of your folder for testing
        callbacks=[PhysicsLossWarmup()],
    )
    # Save the entire model to a HDF5 file or SavedModel format
    model.save("100000k_hydrology_deeplab_lstm_model.keras")

    MODEL_FILE = r'/Weights/hydrology_deeplab_lstm_model.keras'
    PREDICT_FILE = r'C:\Users\dcost\Chandra Mentorship\Example Dataset\input\Aug_29_2017_73.75.npy'
    TRUE_FILE = r'C:\Users\dcost\Chandra Mentorship\Example Dataset\output\Aug_29_2017_73.75.npy'

    result = compare_prediction(MODEL_FILE, PREDICT_FILE,TRUE_FILE)
