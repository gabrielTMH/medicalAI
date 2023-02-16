import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#look into messing around with these numbers
max_features=100
max_len=10
def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].to_numpy(), dataframe["target"].to_numpy())
    )
    return dataset


dataFrame = pd.read_csv("data/Medical_Error_Test_Data.csv")
for column in dataFrame.columns:
    dataFrame[column] = dataFrame[column].replace(np.nan, ' ')
dataFrame['text'] = dataFrame['Interlock 1'] + ' ' + dataFrame['Interlock 2'] + ' ' + dataFrame['Interlock 3'] + ' ' + dataFrame['Error Code 1'] + ' ' + dataFrame['Error Code 2']
dataFrame['target'] = dataFrame['Resolution 1'] + ' ' + dataFrame['Resolution 2'] + ' ' + dataFrame['Resolution 3']
df_shuffled = dataFrame.sample(frac=1, random_state=42)
print(df_shuffled.head())
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)
print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")
vectorize_layer = tf.keras.layers.TextVectorization(
 max_tokens=max_features,
 output_mode='int',
 output_sequence_length=max_len)

full_ds=create_dataset(df_shuffled)
print(full_ds)

vectorize_layer.adapt(create_dataset(df_shuffled))

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
model.add(vectorize_layer)
input_data = [["UDRS"], ["Leaf Stall"],['FLOW	PUMP	KFIL']]
predictions = model.predict(input_data)
print(predictions)