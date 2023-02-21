import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from tensorflow import keras


# look into messing around with these numbers
max_features = 100
max_len = 10


def create_dataFrame_from_csv(name):
    df = pd.read_csv(name)
    #todo read in data in a way that can be used by a decision forest and is also mutable
    dataset = tfdf.keras.pd_dataframe_to_tf_dataset(df)
    # dataset = tf.data.experimental.CsvDataset(filenames=name)
    return dataset


def create_dataset(dataframe):
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["text"].to_numpy(), dataframe["target"].to_numpy())
    )
    return tf.reshape(dataset, [1, ])


def clean_data(raw_dataFrame):
    raw_dataFrame['text'] = raw_dataFrame['Interlock 1'] + ' ' + raw_dataFrame['Interlock 2'] + ' ' + raw_dataFrame[
        'Interlock 3'] + ' ' + \
                            raw_dataFrame['Error Code 1'] + ' ' + raw_dataFrame['Error Code 2']
    raw_dataFrame['target'] = raw_dataFrame['Resolution 1']
    return raw_dataFrame


def vectorize_data_set(data_set):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=max_len)
    vectorizer.adapt(data_set)
    vectorized_data_set = vectorizer(data_set)
    print("function ran")
    return vectorized_data_set


def shuffle_and_split(cleaned_dataFrame):
    df_shuffled = cleaned_dataFrame.sample(frac=1, random_state=42)
    test_df = df_shuffled.sample(frac=0.1, random_state=42)
    train_df = df_shuffled.drop(test_df.index)
    return test_df, train_df


dataFrame = create_dataFrame_from_csv("../../data/Medical_Error_Test_Data.csv")
raw_data = create_dataset(dataFrame)
cleaned_data = clean_data(raw_data)
vectorized_data = vectorize_data_set(cleaned_data)
test_ds, train_ds = shuffle_and_split(vectorized_data)
print(vectorized_data)

