from codecarbon import EmissionsTracker



import tensorflow as tf
from tensorflow import keras
import numpy as np



try:
	tracker = EmissionsTracker(project_name="MI_PROYECTO", measure_power_secs=10)
	
#########################################################################################

	tracker.start_task("CARGAMOS DATASET")
	# 1. Cargar y preprocesar datos
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

	# Normalizar y redimensionar
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0
	x_train = x_train.reshape(-1, 28, 28, 1)
	x_test = x_test.reshape(-1, 28, 28, 1)

	# Convertir etiquetas a one-hot encoding
	y_train_cat = keras.utils.to_categorical(y_train, 10)
	y_test_cat = keras.utils.to_categorical(y_test, 10)

	emisiones_carga = tracker.stop_task()

#########################################################################################

	tracker.start_task("CONSTRUIR MODELO")
	# 2. Crear modelo
	model = keras.Sequential([
		keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
		keras.layers.MaxPooling2D((2,2)),
		keras.layers.Flatten(),
		keras.layers.Dense(128, activation='relu'),
		keras.layers.Dense(10, activation='softmax')
	])

	# 3. Compilar y entrenar
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.2)
	
	emisiones_modelo = tracker.stop_task()

#########################################################################################
	
	tracker.start_task("EVALUAR MODELO")
	# 4. Evaluar
	test_loss, test_acc = model.evaluate(x_test, y_test_cat)
	print(f"\nPrecisión en test: {test_acc:.4f}")
	emisiones_evaluacion = tracker.stop_task()

#########################################################################################

	tracker.start_task("INFERENCIA")

	# 5. Hacer inferencia
	predictions = model.predict(x_test)
	predicted_classes = np.argmax(predictions, axis=1)
	# Mostrar algunas predicciones (de las 10K)
	print("\nPrimeras 100 predicciones:")
	for i in range(100):
		print(f"Real: {y_test[i]}, Predicho: {predicted_classes[i]}")
	emisiones_inferencia = tracker.stop_task()

#########################################################################################

finally:
	_ = tracker.stop()
