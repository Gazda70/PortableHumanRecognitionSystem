import tensorflow as tf  # now import the tensorflow module
print(tf.version)  # make sure the version is 2.x

my_first_tensor = tf.Variable([[[1, 2, 3], [7, 8, 9]], [[56, 67, 54], [1, 0, 1]]], tf.int32)
my_second_tensor = tf.Variable([[4,5,6],[7, 8, 9]], tf.int32)

first_reshape = tf.reshape(my_first_tensor, [4, 3])
second_reshape = tf.reshape(my_second_tensor, [3, 2])

print("\n\n")
print("First version of tensors\n")
print(tf.rank(my_first_tensor))
print(my_first_tensor)
print(tf.rank(my_second_tensor))
print(my_second_tensor)
print("\n\n")
print("Tensors after being reshaped\n")
print(tf.rank(first_reshape))
print(first_reshape)
print(tf.rank(second_reshape))
print(second_reshape)

print(my_first_tensor[:, :, :1])

tensor_crazy = tf.ones([5, 5, 5, 5])

tensor_crazy = tf.reshape(tensor_crazy, [25, 25])
print(tensor_crazy)
