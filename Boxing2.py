import gymnasium as gym
import numpy as np

# Crear el ambiente
env = gym.make("ALE/Boxing-ram-v5")

Q = np.zeros([env.action_space.n, env.observation_space.shape[0]])

# Establecemos los parámetros de aprendizaje
alpha = 0.1
gamma = 0.6
epsilon = 0.1
num_episodios = 2000

# Definimos una función auxiliar para convertir el estado en un array

def estado_array(estado):
    print(estado)
    print('-------------------')
    estado = np.array(estado).flatten()
    estado_reshaped = estado.reshape(1, -1)
    return np.ravel_multi_index(estado_reshaped[0], dims=env.observation_space.shape)


# Comenzamos el entrenamiento
for episodio in range(num_episodios):
    estado = env.reset()
    # estado_idx = estado_array(estado)
    estado_list = list(estado[0])
    estado_list.append(estado[1]['lives'])  # Agrega el valor de 'lives' al final de la lista
    estado_idx = np.ravel_multi_index(tuple(estado_reshaped[0]), dims=env.observation_space.shape)
    print(estado)
    done = False
    
    while not done:
        # Escogemos una acción e-greedy
        if np.random.uniform(0, 1) < epsilon:
            accion = env.action_space.sample()
        else:
            accion = np.argmax(Q[int(estado_idx), :])
        
        # Ejecutamos la acción y obtenemos la siguiente observación y la recompensa
        siguiente_estado, recompensa, done, info, prob = env.step(accion)
        siguiente_estado_idx = estado_array(siguiente_estado)
        
        # Actualizamos la tabla Q
        Q[accion, estado_idx] += alpha * (recompensa + gamma * np.max(Q[:, siguiente_estado_idx]) - Q[accion, estado_idx])
        
        estado_idx = siguiente_estado_idx
    
    # Imprimimos el progreso
    if (episodio+1) % 100 == 0:
        print("Episodio {} completado".format(episodio+1))
        
#Indicar que termino
print("Entrenamiento completado")

# Evaluar el agente entrenado
estado = env.reset()
estado = estado[0]
done = False
total_reward = 0

while not done:
    # Elegir la mejor acción
    action = np.argmax(Q[int(estado)])
    
    # Realizar la acción y obtener la recompensa y el siguiente estado
    next_state, reward, done, info = env.step(action)
    
    # Actualizar el estado actual y la recompensa total
    state = next_state
    total_reward += reward
    
print("Total reward: {}".format(total_reward))

# cerrar el entorno
env.close()
