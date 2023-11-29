# Repositorio DRL
Este repositorio contiene los archivos para realizar un entrenamiento de agentes DQN en un entorno personalizado para toma de decisiones forestales (construcción de caminos y cosecha de rodales) acotadas al caso de estudio del predio Los Copihues. Se trata de un problema determinista, entonces existe una base de datos donde se almacenan los parámetros necesarios para resolver el problema de optimización. 

  El archivo AgenteDQN+DNN contiene las clases de agente y de red neuronal para DQN.

  El archivo Entorno_LosCopihues contiene la clase que simula las transiciones y recompensas de la administración del bosque.

  El archivo MODELO_MIP_V5 fixed es la resolución óptima del problema mediante programación entera mixta.
