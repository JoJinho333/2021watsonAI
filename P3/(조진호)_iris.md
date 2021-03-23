from sklearn.datasets import load_iris iris=load_iris() 

X=tf.constant(iris.data[:,[0,1,2]],dtype=tf.float32) 
< 설명1 : iris.data의 0,1,2번째 값인 SepalLengthCm, SepalWidthCm, PetalLengthCm를 입력값 x로 하고, 상수의 데이터형은 tf.float32 >

y=tf.constant(iris.data[:,3],dtype=tf.float32)

w=tf.Variable(tf.random.normal([3,5])) 
b=tf.Variable(tf.random.normal([5]))

u=tf.nn.relu(X@w+b)
< 설명2 : 인풋값 x, 가중치 w, 상수항 b를 사용해서 활성화 함수 ReLU를 적용시키고 hidden layer 활성화 >

ww=tf.Variable(tf.random.normal([5,5])) 
bb=tf.Variable(tf.random.normal([5]))

uu=tf.nn.relu(u@ww+bb)

www=tf.Variable(tf.random.normal([5,1])) 
bbb=tf.Variable(tf.random.normal([]))

pred_y=uu@www+bbb 
< 설명3 : 중간층을 2번 지나면서 PetalWidthCm에 대한 예측값 >

mse=tf.reduce_mean(tf.square(y-pred_y)) 
< 설명4 : 평균제곱오차인 mse에 실제값 y와 예측값 pred_y의 차인 오차 제곱의 평균 (값이 크면 정답에서 멀어진 것이고, 값이 작으면 예측된 값이 정답에 더 가까운 것이라 판단) > 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001) 
train_op=optimizer.minimize(mse) 
< 설명5 : 오차를 최소화시키기 위해 learning_rate(학습률)을 0.001로 설정한 optimizer(경사하강법)의 최소값을 저장 >

costs=[]

tf.global_variables_initializer().run()
