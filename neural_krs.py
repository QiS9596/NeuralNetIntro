import dataGenerator
import NeuralNets as NN
import pandas as pd
import numpy as np

dg = dataGenerator.ParityDataGenerator()
df = dg.generate_data()

from keras.layers import Dense, Input
from keras.models import Model
from keras import optimizers
from keras import initializers
df = df.sample(frac=1.0)
input = Input([4, ])
initializer = initializers.RandomUniform(minval=-1.0, maxval=1)
d = Dense(4, activation='sigmoid',kernel_initializer=initializer)(input)
# d = Dense(4, activation='sigmoid')(d)
o = Dense(1, activation='sigmoid', kernel_initializer=initializer)(d)
optimizer = optimizers.SGD(lr=5e-2, momentum=0)
model = Model(inputs=input, outputs=o)
model.compile(optimizer, loss='mse', metrics=['acc'])
model.fit(df.drop(columns=['label']).values, df['label'].values, epochs=100000,  batch_size=1,verbose=1)
print(model.evaluate(df.drop(columns=['label']).values, df['label'].values))