import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].to_numpy(), dataframe["target"].to_numpy())
    )
    # dataset = dataset.batch(100)
    # dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


dataFrame = pd.read_csv("data/Medical_Error_Test_Data.csv")
for column in dataFrame.columns:
    dataFrame[column] = dataFrame[column].replace(np.nan, ' ')
dataFrame['text'] = dataFrame['Interlock 1'] + ' ' + dataFrame['Interlock 2'] + ' ' + dataFrame['Interlock 3'] + ' ' + dataFrame['Error Code 1'] + ' ' + dataFrame['Error Code 2']
dataFrame['target'] = dataFrame['Resolution 1'] + ' ' + dataFrame['Resolution 2'] + ' ' + dataFrame['Resolution 3']
df_shuffled = dataFrame.sample(frac=1, random_state=42)
df_shuffled.drop(["Problem", "Sub System", "Interlock 4", "Interlock 5","Interlock 1","Interlock 2","Interlock 3","Error Code 1","Error Code 2","Error Code 3"], axis=1, inplace=True)
df_shuffled.reset_index(inplace=True, drop=True)
print(df_shuffled.head())
test_df = df_shuffled.sample(frac=0.1, random_state=42)
train_df = df_shuffled.drop(test_df.index)
print(f"Using {len(train_df)} samples for training and {len(test_df)} for validation")

max_features=100
max_len=20

vectorize_layer = tf.keras.layers.TextVectorization(
 max_tokens=max_features,
 output_mode='int',
 output_sequence_length=max_len)

full_ds=create_dataset(df_shuffled)
print(full_ds)

vectorize_layer.adapt(create_dataset(df_shuffled))
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
# model.add(vectorize_layer)
# input_data = [["UDRS"], ["Leaf Stall"]]
# model.predict(input_data)