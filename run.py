import numpy as np
import tensorflow as tf
from preprocess import import_data, get_next_batch
from cnn_encoder_model import CNNAutoEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def train(model, geometries, results):
	completed = 0

	ux_loss_list = []
	uy_loss_list = []
	p_loss_list = []

	while completed < len(geometries):

		our_size = model.batch_size

		if len(geometries) - completed < model.batch_size:
			our_size = len(geometries) - completed

		geom_batch, results_batch = get_next_batch(geometries, results, completed, our_size)

		completed += our_size

		with tf.GradientTape() as tape:
			# ux, uy, p = model.call(geom_batch)
			ux = model.call(geom_batch)
			ux_loss = model.loss_function(ux, results_batch[:,0,:,:],geom_batch) #geometries serves as the mask here
			# uy_loss = model.loss_function(uy, results_batch[:,1,:,:],geom_batch)
			# p_loss = model.loss_function(p, results_batch[:,2,:,:],geom_batch)

		# gradients = tape.gradient([ux_loss, uy_loss, p_loss], model.trainable_variables)
		gradients = tape.gradient([ux_loss], model.trainable_variables)
		model.adam_optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		ux_loss_list.append(ux_loss)
		# uy_loss_list.append(uy_loss)
		# p_loss_list.append(p_loss)

	print('Ux loss: ', tf.reduce_mean(ux_loss_list))
	# print('Uy loss: ', tf.reduce_mean(uy_loss_list))
	# print('P loss: ', tf.reduce_mean(p_loss_list))

def test(model, geometries, results):
	completed = 0

	ux_loss_list = []
	uy_loss_list = []
	p_loss_list = []

	while completed < len(geometries):

		our_size = model.batch_size

		if len(geometries) - completed < model.batch_size:
			our_size = len(geometries) - completed

		geom_batch, results_batch = get_next_batch(geometries, results, completed, our_size)

		completed += our_size

		# ux, uy, p = model.call(geom_batch)
		ux = model.call(geom_batch)

		ux_loss = model.loss_function(ux, results_batch[:,0,:,:],geom_batch) #geometries serves as the mask here
		# uy_loss = model.loss_function(uy, results_batch[:,1,:,:],geom_batch)
		# p_loss = model.loss_function(p, results_batch[:,2,:,:],geom_batch)

		ux_loss_list.append(ux_loss)
		# uy_loss_list.append(uy_loss)
		# p_loss_list.append(p_loss)
	print(' ')
	print('Testing Results:')
	print('Ux loss: ', tf.reduce_mean(ux_loss_list))
	# print('Uy loss: ', tf.reduce_mean(uy_loss_list))
	# print('P loss: ', tf.reduce_mean(p_loss_list))
	print(' ')

def main():

	geometries, outputs = import_data()

	geom_train, geom_test, output_train, output_test = train_test_split(geometries, outputs, test_size=0.2)

	Our_Model = CNNAutoEncoder()

	for i in np.arange(Our_Model.epochs):
		print(i)
		train(Our_Model, geom_train, output_train)

	test(Our_Model, geom_test, output_test)

	random = np.random.randint(0,geometries.shape[0])

	ux1 = Our_Model.call(tf.expand_dims(geometries[random],0))
	ux1 = tf.squeeze(ux1)
	# uy1 = tf.squeeze(uy1)
	# p1 = tf.squeeze(p1)

	trux = tf.squeeze(outputs[random][0,:,:])
	# truy = tf.squeeze(outputs[random][1,:,:])
	# trup = tf.squeeze(outputs[random][2,:,:])

	testux1 = tf.expand_dims(ux1,0)
	# testtrux = tf.expand_dims(trux,0)
	# testgeom = tf.expand_dims(geometries[random],0)

	# print(testux1.shape)
	# print(testtrux.shape)
	# print(testgeom.shape)
	#
	#
	# print(Our_Model.loss_function(testux1,testtrux,testgeom))

	fig = plt.figure()
	ax = fig.subplots(1,2)
	bo = ax[0].imshow(tf.squeeze(ux1))
	ax[0].set_title('Predicted')
	ax[0].set_ylabel('U_x')
	fig.colorbar(bo, ax=ax[0])
	# ax[1,0].imshow(uy1)
	# ax[1,0].set_ylabel('U_y')
	# ax[2,0].imshow(p1)
	# ax[2,0].set_ylabel('P')
	yo = ax[1].imshow(trux)
	ax[1].set_title('CFD Results')
	fig.colorbar(yo, ax=ax[1])
	# ax[1,1].imshow(truy)
	# ax[2,1].imshow(trup)
	fig.savefig('results.png')

	print(ux1)
	tf.io.write_file('testt',ux1)


	pass

if __name__ == '__main__':
	main()
