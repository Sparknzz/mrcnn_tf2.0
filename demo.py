import tensorflow as tf

random_float = tf.random.uniform(shape=())
print(random_float)

A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
D = tf.matmul(A, B)  # 计算矩阵A和B的乘积
print(D)

# autograd
x = tf.Variable(initial_value=3.)
# tf.GradientTape 自动求导记录器　在这其中的步骤与变量都会被记录
with tf.GradientTape() as tape:
    y = tf.square(x)  # y = x**2

y_grad = tape.gradient(y, x)

print([y, y_grad])

#############################################################################################
# L(w,b) = |Xw + b - y|**2 求ｗ　ｂ的偏导
#############################################################################################
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
x = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])

with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x, w) + b - y))

w_grad, b_grad = tape.gradient(L, [w, b])
print(L.numpy(), w_grad, b_grad)


##########################################################################################
##########################################################################################
# 使用keras model layers 实现线性回归
##########################################################################################
##########################################################################################
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


model = Linear()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.reduce_mean(tf.square(y_pred - y))

    grads = tape.gradient(loss, model.varaibles)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)