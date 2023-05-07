import gymnasium as gym
import numpy as np
import tensorflow as tf

# Crear el entorno del Boxeo
env = gym.make("ALE/Boxing-ram-v5", difficulty=2)

# Crear una red neuronal con una capa oculta de 64 neuronas
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(env.observation_space.shape[0],)),
    tf.keras.layers.Dense(env.action_space.n, activation='linear')
])

# Compilar el modelo con el optimizador Adam y la función de pérdida de error cuadrático medio
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Entrenar el modelo
episodes = 20
steps = 200
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
gamma = 0.89

for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state[0], [1, env.observation_space.shape[0]])

    done = False
    total_reward = 0
    
    for step in range(steps):
        # Elegir una acción con un valor epsilon-greedy
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state, verbose=0)[0])
        
        # Tomar la acción y obtener la siguiente observación, recompensa y estado de finalización
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        
        # Calcular la recompensa total y ajustar la red neuronal
        total_reward += reward
        target = reward + gamma * np.amax(model.predict(next_state, verbose=0)[0])
        target_f = model.predict(state, verbose=0)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
        
        # Actualizar el estado actual y la variable epsilon
        state = next_state
        if done:
            print("Episodio {}, Recompensa total {}".format(episode, total_reward))
            break
    # Decrementar el valor de epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env = gym.make("ALE/Boxing-ram-v5", render_mode="human")
# correr un episodio
state = env.reset()
state = np.reshape(state[0], [1, env.observation_space.shape[0]])

done = False
while not done:
    # elegir una acción
    action = np.argmax(model.predict(state, verbose = 0)[0])
    
    # tomar la acción en el ambiente de juego
    next_state, reward, done, info, _ = env.step(action)
    next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
    
    # mostrar el ambiente de juego
    env.render()
    
    # actualizar el estado actual
    state = next_state