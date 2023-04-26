import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# hiperparámetros del algoritmo
alpha = 0.1    # tasa de aprendizaje
gamma = 0.99   # factor de descuento
epsilon = 0.1  # probabilidad de exploración

# inicializar la tabla Q
Q = np.zeros((env.observation_space.n, env.action_space.n))

# número de episodios de entrenamiento
num_episodes = 10000

# entrenamiento del agente
for episode in range(num_episodes):
    # resetear el entorno al inicio de cada episodio
    state = env.reset()
    
    # loop de ejecución del episodio
    done = False
    while not done:
        # elegir la siguiente acción utilizando una política e-greedy
        if np.random.rand() < epsilon:
            # acción aleatoria con probabilidad epsilon
            action = env.action_space.sample()
        else:
            # acción greedy con probabilidad 1-epsilon
            action = np.argmax(Q[int(state)])
            
        # ejecutar la acción y obtener la siguiente tupla (estado, recompensa, done, info)
        next_state, reward, done, info = env.step(action)
        
        # actualizar la tabla Q utilizando la ecuación de Q-learning
        Q[int(state), int(action)] += alpha * (reward + gamma * np.max(Q[int(next_state)]) - Q[int(state), int(action)])
        
        # actualizar el estado actual
        state = next_state
        
    # imprimir el progreso del entrenamiento
    if (episode+1) % 100 == 0:
        print(f'Episode {episode+1}/{num_episodes}')
    
# cerrar el entorno
env.close()

# ejecutar el entorno con la política óptima
state = env.reset()
done = False
while not done:
    action = np.argmax(Q[int(state)])
    state, _, done, _ = env.step(action)
    env.render()

# cerrar el entorno
env.close()