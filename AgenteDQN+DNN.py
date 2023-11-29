import random
from ClaseBosque_9 import Bosque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import json
import math
import pickle

#Inicializar semilla en distintas librerías para asegurar reproducción de los datos  
seed_value = 17  
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

#Clase red neuronal
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, action_size)
        self.fc2 = nn.Linear(action_size, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 25)
        self.fc5 = nn.Linear(25, 25)
        self.fc6 = nn.Linear(25, 25)
        self.fc7 = nn.Linear(25, 25)
        self.fc8 = nn.Linear(25, action_size)
        self.fc9 = nn.Linear(action_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        return self.fc9(x)

#Clase agente DQN A
class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.004,
                 gamma=0.6, buffer_size=512, batch_size=128, epsilon_start=1.0,
                 epsilon_end=0, epsilon_decay=0.995, update_freq=33, tau=0.2):
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_freq = update_freq
        self.tau = tau
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=buffer_size)
        #self.memory = loaded_d
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_step_counter = 0
        self.loss = []
        self.q_promedio = []
	
    def choose_action(self, state, epsilon):
        #state_array = self.state_to_tensor(state)
        if np.random.random() < epsilon:
            print("epsilon mayor a numero aleatorio -> ACCION ALEATORIA")
            print(state)
            # Solo considera las acciones que están disponibles
            available_actions = np.where(state[:44] == 0)[0]

            # Si se cumple la condición para la última acción, la añade a las acciones disponibles
            if state[47] == 1:
                available_actions = np.append(available_actions, 44)
                
            action = np.random.choice(available_actions)
            print(available_actions,action)
        else:
            print("Epsilon menor a valor aleatorio -> MEJOR ACCION CONOCIDA")
            print(state)
            state_tensor = torch.FloatTensor(state).to(self.device)
            print("El tensor está en GPU? = ", state_tensor.is_cuda)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            
            # Establece los valores Q de las acciones no disponibles a -infinito
            q_values[np.where(state[:44] != 0)[0]] = -np.inf

            # Si no se cumple la condición para la última acción, la excluye
            if state[47] == -1:
                q_values[-1] = -np.inf 
            
            action = torch.argmax(q_values).item()
            print(q_values,action)

        if self.learning_step_counter % self.update_freq == 0:
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values=self.q_net(state_tensor)
            q_values[np.where(state[:44] != 0)[0]] = -np.inf

            # Si no se cumple la condición para la última acción, la excluye
            if state[47] == -1:
                q_values[-1] = -np.inf 
            non_inf_indices = torch.where(q_values != float('-inf'))[0]

            # Usa esos índices para extraer los elementos que son diferentes de -inf
            non_inf_values = q_values[non_inf_indices]

            # Calcula la media de esos valores
            if len(non_inf_values) > 0:  # Evita dividir por cero
                mean_value = torch.mean(non_inf_values).item()
            else:
                mean_value = None  # o cualquier valor que desees asignar en este caso

            # Almacena el valor en un vector de Python, si así lo deseas
            self.q_promedio.append(mean_value)
            
        return action

    
        
    def step(self, state, action, reward, next_state, done, t):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        if isinstance(state, np.ndarray):
            next_state = torch.from_numpy(next_state).float().to(self.device)    

        self.memory.append((state, action, float(reward), next_state, done)) #recompensa como float

        if len(self.memory) >= self.batch_size:
            self._learn()

        if done:
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)

    def _learn(self):
        states, actions, rewards, next_states, dones = self._sample_memory()

        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        q_expected = self.q_net(states).gather(1, actions)

        loss = nn.functional.mse_loss(q_expected, q_targets)
        #loss = nn.functional.smooth_l1_loss(q_expected, q_targets)
        #self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learning_step_counter += 1

        if self.learning_step_counter % self.update_freq == 0:
            self.loss.append(loss.item())
            self._update_target_net()

    def _sample_memory(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0].cpu() if torch.is_tensor(e[0]) else e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2].cpu() if torch.is_tensor(e[2]) else e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3].cpu() if torch.is_tensor(e[3]) else e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4].cpu() if torch.is_tensor(e[4]) else e[4] for e in experiences])).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def _update_target_net(self):
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

#Funciones auxiliares
def state_to_array(state):                                                                          ##
        rodales_y_caminos = np.array(state[0]).flatten()  # Asegúrate de que esto es un array 1D    ##
        periodo = np.array([state[1]])                                                              ##
        volumen_acumulado = np.array([state[2]])                                                    ##
        demanda = np.array([state[3]])                                                              ##
        check = np.array([state[4]])         
        cortados = np.array([state[5]])                                                       ##
        ESTADO = np.concatenate([rodales_y_caminos, periodo, volumen_acumulado, demanda,check,cortados])
        return normalizar(ESTADO)        ##


def normalizar(state):
    state[44] = (state[44]-2)/(7-2)
    state[45] = state[45]/6500
    state[46] = (state[46]-1304)/(1346-1304)
    state[48] = state[48]/25
    return state

#Carga de entorno y agente A

env = Bosque()  
state_size = 49
action_size = 45

agent = DQNAgent(state_size, action_size)

#ENTRENAMIENTO
num_episodes = 1000 #18:15 - XX:XX
rewards = []
epsilons = []
successful_episode_actions = {}
RECOMPENSAS = {}
RECOMPENSAS_TEST = {}
PERDIDA = {}
Q_VALUES = {}
largo_episodios = []

for i_episode in range(1,1+num_episodes):
    state = env.reset()
    state = state_to_array(state)
    total_reward = 0
    actions_taken = []
    if i_episode <= 0:### si es cero, entonces ya no se ejecuta esta secuencia de acciones
        for t in range(len(acciones_pregrabadas[i_episode])):
            print(t,int(acciones_pregrabadas[i_episode][t]))
            ACCION=int(acciones_pregrabadas[i_episode][t])
            next_state, reward, done, _ = env.step(ACCION)
            print(reward)
            next_state = state_to_array(next_state)
            agent.step(state, ACCION, reward, next_state, done, t)
            state = next_state
            total_reward += reward
            print(total_reward)
            actions_taken.append(ACCION)

            if done:
                print(f"Episodio {i_episode} finalizado después de {t+1} pasos, recompensa total: {total_reward}")
                #GUARDO LAS RECOMPENSAS DE TODOS LOS EPISODIOS FINALIZADOS
                RECOMPENSAS[i_episode] = total_reward
                if state[44] == 1 and state[47] == 1:
                    actions_taken.append(total_reward)
                    successful_episode_actions[i_episode] = actions_taken
                break
    else:
        #Exploración normal
        #with open('my_deque.pkl', 'wb') as f:
            #pickle.dump(agent.memory, f)
        for t in range(1000):  # Reemplaza 1000 por el número máximo de pasos si existe
            action = agent.choose_action(state, agent.epsilon)
            actions_taken.append(action)
            next_state, reward, done, _ = env.step(action)
            print(reward)
            next_state = state_to_array(next_state)
            agent.step(state, action, reward, next_state, done, t)
            state = next_state
            total_reward += reward
            print(total_reward)
            print("A continuación como se ve el estado luego del loop de entrenamiento: ",state)
            
            if done:
                print(f"Episodio {i_episode} finalizado después de {t+1} pasos, recompensa total: {total_reward}")
                #GUARDO LAS RECOMPENSAS DE TODOS LOS EPISODIOS
                RECOMPENSAS[i_episode] = total_reward
                largo_episodios.append((t,state[44],state[47]))#el paso , el periodo en que acabó , si se satisface la demanda
                if state[44] == 1 and state[47] == 1:
                    successful_episode_actions[i_episode] = actions_taken
                    actions_taken.append(total_reward)
                    
                break
    rewards.append(total_reward)
    epsilons.append(agent.epsilon)
    
PERDIDA[3] = agent.loss
Q_VALUES[3] = agent.q_promedio

successful_episode_actions = {k: [int(a) for a in v] for k, v in successful_episode_actions.items()}
with open('C3_ACCIONES_EXITOSAS_DQN.json', 'w') as archivo:
    json.dump(successful_episode_actions, archivo)
    print('Archivo exportado con exito')

with open('C3_REW_TRAIN_DQN.json', 'w') as archivo:
    json.dump(RECOMPENSAS, archivo)
    print('Archivo exportado con exito')

with open('C3_PERDIDA_DQN.json', 'w') as archivo:
    json.dump(PERDIDA, archivo)
    print('Archivo exportado con exito')

with open('C3_QVALUES_DQN.json', 'w') as archivo:
    json.dump(Q_VALUES, archivo)
    print('Archivo exportado con exito')

with open('C3_largoepisodios_DQN.json', 'w') as archivo:
    json.dump(largo_episodios, archivo)
    print('Archivo exportado con exito')

#GUARDAR PESOS DE LA RED
torch.save(agent.q_net.state_dict(), 'C3_PESOS_DQN.pth')


# Ahora puedes trazar las recompensas y los valores de épsilon a lo largo del tiempo
plt.figure(figsize=(12,6))
plt.subplot(2, 1, 1)
plt.plot(rewards)
plt.title('Recompensa por episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')

plt.subplot(2, 1, 2)
plt.plot(epsilons)
plt.title('Epsilon por episodio')
plt.xlabel('Episodio')
plt.ylabel('Epsilon')

plt.tight_layout()
plt.show()

#PROMEDIO MOVIL ENTRENAMIENTO
window_size = int(30)
smoothed_rewards = [np.mean(rewards[max(0, i-window_size):(i+1)]) for i in range(len(rewards))]

plt.figure(figsize=(12,6))
plt.plot(smoothed_rewards)
plt.title('Promedio móvil de recompensa por episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Promedio Móvil')
plt.show()

#solo el periodo en que terminó
T=[]
for i in largo_episodios:
    T.append(i[0])

plt.figure(figsize=(12,6))
plt.plot(T)
plt.title('Cantidad de pasos por episodio')
plt.xlabel('Episodio')
plt.ylabel('Largo del episodio')
plt.show()

plt.figure(figsize=(12,6))
plt.plot(agent.q_promedio)
plt.title('Promedio de Q-value cada N = '+str(agent.update_freq)+' pasos')
plt.xlabel('Episodio')
plt.ylabel('Q-value promedio')
plt.show()

plt.figure(figsize=(12,6))
plt.plot(agent.loss)
plt.title('Valor función de pérdida cada N = '+str(agent.update_freq)+' pasos')
plt.xlabel('Episodio')
plt.ylabel('Valor función de pérdida')
plt.show()
#CARGAR ENTRENAMIENTO ANTERIOR
# Cargar los pesos del archivo
#agent.q_net.load_state_dict(torch.load('Q_network_checkpoint_500_30n_2e-3.pth'))

# Si estás en modo de inferencia (es decir, no entrenamiento), también deberías llamar a .eval()
#agent.q_net.eval()
#TESTEO
num_test_episodes = 3
test_rewards = []

for i_episode in range(num_test_episodes):
    state = env.reset()
    state = state_to_array(state)
    total_reward = 0
    for t in range(1000):  # De nuevo, reemplaza 1000 por el número máximo de pasos si lo tienes
        action = agent.choose_action(state, 0)  # No exploramos mucho
        next_state, reward, done, _ = env.step(action)
        next_state = state_to_array(next_state)
        total_reward += reward
        state = next_state

        if done:
            print(f"Episodio de prueba {i_episode} finalizado después de {t+1} pasos, recompensa total: {total_reward}")
            #GUARDO LAS RECOMPENSAS DE TODOS LOS EPISODIOS
            RECOMPENSAS_TEST[i_episode] = total_reward
            break
    test_rewards.append(total_reward)


with open('C3_REWARD_TEST_DQN.json', 'w') as archivo:
    json.dump(RECOMPENSAS_TEST, archivo)
    print('Archivo exportado con exito')
    
 #Ahora puedes trazar las recompensas del testeo a lo largo del tiempo
plt.figure(figsize=(12,6))
plt.plot(test_rewards)
plt.title('Recompensa por episodio de prueba')
plt.xlabel('Episodio de prueba')
plt.ylabel('Recompensa')
plt.show()

# Y también el promedio móvil de las recompensas del testeo
window_size = int(5)  # puedes ajustar este valor
smoothed_test_rewards = [np.mean(test_rewards[max(0, i-window_size):(i+1)]) for i in range(len(test_rewards))]

plt.figure(figsize=(12,6))
plt.plot(smoothed_test_rewards)
plt.title('Promedio móvil de recompensa por episodio de prueba')
plt.xlabel('Episodio de prueba')
plt.ylabel('Recompensa Promedio Móvil')
plt.show()

#02:07 - 15:40
