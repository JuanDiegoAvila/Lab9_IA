import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np


env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# ACTION SPACE 0: left, 1: down, 2: right, 3: up
# OBSERVATION SPACE 16: 4x4 grid
# REWARD 0: hole, 1: goal, 0: frozen

# Termina cuando:
# 1. El agente llega al estado (max(nrow) * max(ncol) - 1) o [max(nrow)-1, max(ncol)-1](goal)
# 2. El agente cae en un agujero (hole)

# Q = [[0 for i in range(env.action_space.n)] for j in range(env.observation_space.n)]
# Q = np.array(Q, dtype=np.float32)
Q = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.8
gamma = 0.95
episodios = 1000

for i in range(episodios):
    estado = env.reset()
    estado = estado[0]

    done = False

    while not done:
        # Convertir el estado en un array unidimensional
        estado_array = np.array([estado], dtype=np.int16)

        # Seleccionar una acción a partir del estado actual utilizando la política epsilon-greedy
        if np.random.uniform(0, 1) < 0.5:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[estado_array.item(), :])

        resultados = env.step(a)
        estado2, recompensa, done, info, probabilidad = resultados

        estado_array2 = np.array([estado2], dtype=np.int16)

        Q[estado_array.item(), a] = Q[estado_array.item(), a] + alpha * (recompensa + gamma * np.max(Q[estado_array.item(), :]) - Q[estado_array.item(), a])

        estado = estado2

print(Q)


