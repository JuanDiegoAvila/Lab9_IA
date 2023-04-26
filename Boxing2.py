import gymnasium as gym
import numpy as np

# hiperparámetros
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999

# crear entorno
env = gym.make('ALE/Boxing-v5')

# inicializar tabla Q
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))

# episodios
num_episodes = 10000
max_steps_per_episode = 10000
total_reward = np.zeros(num_episodes)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    t = 0
    episode_reward = 0
    
    while not done and t < max_steps_per_episode:
        # seleccionar acción mediante política epsilon-greedy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample() # exploración
        else:
            action = np.argmax(q_table[state, :]) # explotación
        
        # realizar acción y obtener observación y recompensa
        next_state, reward, done, info = env.step(action)
        
        # actualizar tabla Q
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        # actualizar estado y recompensa acumulada
        state = next_state
        episode_reward += reward
        t += 1
    
    # reducir valor de epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # almacenar recompensa del episodio
    total_reward[episode] = episode_reward
    
    # imprimir progreso
    if episode % 100 == 0:
        avg_reward = np.mean(total_reward[max(0, episode-100):episode+1])
        print(f'Episode {episode}/{num_episodes}, avg reward: {avg_reward:.2f}, epsilon: {epsilon:.2f}')

# cerrar entorno
env.close()