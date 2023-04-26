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
    return np.array(estado).reshape((env.observation_space.shape[0],))

# Comenzamos el entrenamiento
for episodio in range(num_episodios):
    estado = env.reset()
    done = False
    
    while not done:
        # Escogemos una acción e-greedy
        if np.random.uniform(0, 1) < epsilon:
            accion = env.action_space.sample()
        else:
            accion = np.argmax(Q[:, estado_array(estado).item()])
        
        # Ejecutamos la acción y obtenemos la siguiente observación y la recompensa
        siguiente_estado, recompensa, done, info = env.step(accion)
        
        # Actualizamos la tabla Q
        Q[accion, estado_array(estado).item()] += alpha * (recompensa + gamma * np.max(Q[:, estado_array(siguiente_estado).item()]) - Q[accion, estado_array(estado).item()])
        
        estado = siguiente_estado
    
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
