import tensorflow as tf

dummy_data = {
    "names": ["A", "B", "C", "D", "E"],
    "age": [23, 43, 31, 55, 64],
    "weight": [55, 49, 76, 83, 90]
}

datagen = tf.data.Dataset.from_tensor_slices(dummy_data)
datagen = datagen.shuffle(5)
datagen = datagen.batch(2)

for sample in datagen.take(5):
    print("#"*10)
    for key, val in sample.items():
        print(key, val)
