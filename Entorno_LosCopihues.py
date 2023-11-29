import numpy as np
import networkx as nx
import torch as T
import json


class Bosque():
    def __init__(self):
        with open('base_de_datos_E1.json','r') as archivo:
            DATOS= json.load(archivo)
        self.action_space_size = 45
        self.observation_space = {
            'rodales_y_caminos': {'low': -1, 'high': 1, 'shape': (44,), 'dtype': int},
            'periodo': {'size': 7, 'dtype': int},
            'volumen_acumulado': {'low': 0, 'high': float('inf'), 'shape': (1,), 'dtype': float},
            'demanda': {'low': 0, 'high': float('inf'), 'shape': (1,), 'dtype': float},
            'check': {'low': -1, 'high': 1, 'shape': (1,), 'dtype': int},
            'cortados': {'low': 0, 'high':25, 'shape':(1,), 'dtype': int}
        }
        self.CAMINOS=DATOS['CAMINOS']
        self.origenes_existentes_inicial=DATOS['origenes_existentes']
        self.origenes_existentes=self.origenes_existentes_inicial.copy()
        self.caminos_existentes_inicial=DATOS['caminos_existentes']
        self.caminos_existentes=self.caminos_existentes_inicial.copy()
        self.caminos_posibles_inicial=DATOS['caminos_posibles']
        self.caminos_posibles=self.caminos_posibles_inicial.copy()
        self.DEMANDA=DATOS['DEMANDA']
        self.distancias=DATOS['distancias']
        self.rod_ady_origenes=DATOS['rod_ady_origenes']
        self.rodales_ady_rod=DATOS['rodales_ady_rod']
        self.costo_transporte=DATOS['costo_transporte']
        self.costo_construccion_camino=DATOS['costo_construccion_camino']
        self.costo_corte=DATOS['costo_corte']
        self.precio_venta=DATOS['precio_venta']
        self.produccion_rodales=DATOS['produccion_rodales']
        self.t=2
        self.COSTO=0
        self.INGRESO=0
        self.VOLUMEN_PRODUCIDO=0
        self.CORTE_ANTERIOR=[]
        self.CORTADOS_TOTALES=[]
        self.CONSTRUIDOS_ANTERIOR=[]
        self.CONSTRUIDOS_TOTAL=[]
        self.CANTIDAD_CAMINOS=len(self.caminos_existentes.keys())+len(self.caminos_posibles.keys())#
        self.CANTIDAD_RODALES=len(self.rodales_ady_rod.keys())#
        self.done=False
        self.POSICION_RODAL=DATOS['POSICION_RODAL']
        self.POSICION_CAMINOS=DATOS['POSICION_CAMINOS']
        self.total_reward=0
        self.reward=0
        self.prev_reward=0
        # Inicializar estado
        
    def estado(self): #Su única función es devolver el estado en un nuevo formato
        caminos_y_rodales = np.array(self.leer_ambiente()[:44], dtype=int)
        periodo = int(self.leer_ambiente()[44])
        volumen  = float(self.leer_ambiente()[45])
        demanda = float(self.leer_ambiente()[46])
        check = int(self.leer_ambiente()[47])
        cortados = int(self.leer_ambiente()[48])
        return caminos_y_rodales,periodo,volumen,demanda,check,cortados
    
    #MÁSCARA DEL LADO DEL AMBIENTE
    #def get_valid_actions(self):
    #    valid_actions = []
    #    for action_idx in range(self.action_space_size):  # Itera sobre todas las acciones posibles
    #        if self.is_valid_action(action_idx):  # Verifica si la acción es válida
    #            valid_actions.append(action_idx)  # Si la acción es válida, añádela a la lista
    #    return valid_actions

    def actions_space(self): # vector con todas las acciones
        espacio_acciones = []
        for i in range(44): #for i en el espacio de informacion del bosque menos la informacion de periodo y demanda y el volumen
            if self.leer_ambiente()[i]==0:
                espacio_acciones.append(i)
            else:
                espacio_acciones.append(-100)

        if self.leer_ambiente()[47] == 1:
            espacio_acciones.append(44)
        else:
            espacio_acciones.append(-100)
        return espacio_acciones
        
    def available_actions_mask(self): #vector de 0 y 1
        action_mask = []
        for i in range(45):
            if self.actions_space()[i] != -100:
                action_mask.append(1)
            else:
                action_mask.append(0)
        return action_mask
    
    def is_valid_action(self, action):
        if self.available_actions_mask()[action] == 1:
            return False
        else:
            return True
    
    def leer_ambiente(self):
        AMBIENTE_TOTAL=[0 for i in range(5+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS)] #vaciamos el ambiente
        for rodal in self.rodales_ady_rod.keys():
            if self.rodales_existentes() == []:
                if rodal in self.CORTADOS_TOTALES:
                    AMBIENTE_TOTAL[self.POSICION_RODAL[rodal]]+=1
                else:
                    AMBIENTE_TOTAL[self.POSICION_RODAL[rodal]]+=-1
            else:
                if rodal in self.CORTADOS_TOTALES:
                    AMBIENTE_TOTAL[self.POSICION_RODAL[rodal]]+=1
                elif rodal not in self.rodales_existentes():
                    AMBIENTE_TOTAL[self.POSICION_RODAL[rodal]]+=-1
        if self.CONSTRUIDOS_TOTAL==[]:
            for camino in self.caminos_existentes.keys():
                self.CONSTRUIDOS_TOTAL.append(camino)   
        for camino in self.CONSTRUIDOS_TOTAL:
            AMBIENTE_TOTAL[self.POSICION_CAMINOS[camino]]+=1
        for camino in self.caminos_posibles.values():
            if camino not in self.caminos_independientes(self.nodos_independiente()):
                AMBIENTE_TOTAL[self.POSICION_CAMINOS[self.obtener_keys(self.caminos_posibles,camino)]]+=-1
            else:
                pass
        #Agregamos información del volumen y la demanda del mismo periodo
        AMBIENTE_TOTAL[self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= self.t
        AMBIENTE_TOTAL[1+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= self.VOLUMEN_PRODUCIDO 
        AMBIENTE_TOTAL[2+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= int(self.DEMANDA[str(self.t)])
        if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
            AMBIENTE_TOTAL[3+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= 1
        else:
            AMBIENTE_TOTAL[3+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= -1
        AMBIENTE_TOTAL[4+self.CANTIDAD_RODALES+self.CANTIDAD_CAMINOS]+= int(len(self.CORTADOS_TOTALES))
        return AMBIENTE_TOTAL
    
        
    def nodos_independiente(self):
        nodos_independientes=[]
        for i in self.caminos_existentes:
            nodos_independientes.append(self.caminos_existentes[i][0])
            nodos_independientes.append(self.caminos_existentes[i][1])
        nodos_independientes=np.array(nodos_independientes)
        nodos_independientes_unique=np.unique(nodos_independientes)##
        return nodos_independientes_unique
    
    def caminos_independientes(self,NODOS_INDEPENDIENTES):
        caminos_independientes=[]
        for i in range(len(NODOS_INDEPENDIENTES)):
            for j in self.caminos_posibles:
                if self.caminos_posibles[j][0]==NODOS_INDEPENDIENTES[i] or self.caminos_posibles[j][1]==NODOS_INDEPENDIENTES[i]:
                    caminos_independientes.append(self.caminos_posibles[j])
        return caminos_independientes
    
    def obtener_keys(self,DICCIONARIO,VALOR_BUSCADO):
        for i in DICCIONARIO.keys():
            if VALOR_BUSCADO==DICCIONARIO[i]:
                return i
            
    def extraer_origenes(self,listado):
        ORIGENES=[]
        for i in range(len(listado)):
            if listado[i][0]=='o':
                ORIGENES.append(listado[i])
        return ORIGENES

    def quitar_adyacentes(self,X):
        if self.CORTE_ANTERIOR==[]:
            return X
        for j in self.CORTE_ANTERIOR:
            adyacentes=self.rodales_ady_rod[j]
            for ady in adyacentes:
                if ady in X:
                    X.remove(ady)
        return X
    
    def actualizar_origenes_existentes(self):
        for i in self.extraer_origenes(self.nodos_independiente()):
            self.origenes_existentes[i]=self.rod_ady_origenes[i]
            
    def rodales_existentes(self):
        CORTE_PERMITIDO = []
        for i in self.origenes_existentes.values():
            cantidad_rodales_asociados=len(i)
            for j in range(cantidad_rodales_asociados):
                CORTE_PERMITIDO.append(i[j])
        if self.CORTADOS_TOTALES==[]:
            return CORTE_PERMITIDO
        for i in self.CORTADOS_TOTALES:
            if i in CORTE_PERMITIDO:
                CORTE_PERMITIDO.remove(i) #Quita del listado los rodales cortado en etapas anteriores (no se pueden cortar más de 1 vez)
        return self.quitar_adyacentes(CORTE_PERMITIDO)
        
    def construir_camino(self, camino):
        if not self.caminos_posibles:
            print("No hay más caminos para construir")
            return

        # Actualizar caminos existentes y caminos posibles
        nuevo_key = len(self.caminos_existentes) + 1
        self.caminos_existentes[nuevo_key] = self.CAMINOS[camino]
        del self.caminos_posibles[self.obtener_keys(self.caminos_posibles, self.CAMINOS[camino])]

        # Actualizar costos
        costo_construccion = self.costo_construccion_camino[camino][str(self.t)]
        self.COSTO += costo_construccion

        # Actualizar construidos_total y construidos_anterior
        self.CONSTRUIDOS_TOTAL.append(camino)
        self.CONSTRUIDOS_ANTERIOR.append(camino)

        # Actualizar origenes existentes
        self.actualizar_origenes_existentes()

    def cortar_rodal(self, rodal):
        if self.t < 5 and len(self.CORTADOS_TOTALES) > self.CANTIDAD_RODALES // 2: #como en el periodo 1 no se toman decisiones, son 6 los periodos de decisión. A partir del 4to periodo se puede sobrepasar la mitad de la produccion del bosque, es decir, periodo mayores que 4
            self.COSTO+= 10e7
            print("No se puede ejecutar cortar_rodal en este periodo. (SOBREPASARÍAS LA MITAD DEL TOTAL DEL BOSQUE)")
            self.done = True
            return 

        self.CORTADOS_TOTALES.append(rodal)
        self.CORTE_ANTERIOR.append(rodal)

        # Aumentar ingresos y costos
        produccion = self.produccion_rodales[rodal][str(self.t)]
        costo_corte = self.costo_corte[rodal][str(self.t)]
        ingreso = produccion * self.precio_venta[str(self.t)]
        self.INGRESO += ingreso

        # Calcular costo de transporte
        origen = None
        for key, value in self.origenes_existentes.items():
            if rodal in value:
                origen = key
                break

        if origen is not None:
            # Crear un grafo de NetworkX con los caminos existentes
            G = nx.Graph()
            for key, value in self.caminos_existentes.items():
                G.add_edge(value[0], value[1], weight=self.distancias[self.obtener_keys(self.CAMINOS,[value[0],value[1]])])

            # Calcular la distancia más corta desde el origen hasta la salida_1
            distancia = nx.shortest_path_length(G, origen, 'salida_1', weight='weight')

            # Actualizar el costo
            costo_transporte = self.costo_transporte * distancia * produccion
            self.COSTO += costo_corte + costo_transporte
            self.VOLUMEN_PRODUCIDO+= self.produccion_rodales[rodal][str(self.t)]
        else:
            print("No se pudo encontrar el nodo origen para el rodal cortado")
        
        #si corta un rodal y queda en el periodo 3 con 12 rodales cortados, entonces recibe castigo y termina el episodio
        
    def recompensa(self):
        self.total_reward = self.INGRESO-self.COSTO  
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        return self.reward  
        
    def siguiente_periodo(self): #es una tercera accion {ocupa la posición 44}
        RECOMPENSA = 0
        if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
            if self.t==7:
                RECOMPENSA+=1e5
                print("Bien, satisfaces la demanda: volumen = "+str(self.VOLUMEN_PRODUCIDO)+", demanda = "+str(self.DEMANDA[str(self.t)]))
                print("BONO POR TERMINAR CON ÉXITO")
                #print("Utilidad en periodo "+str(self.t)+": "+str(self.recompensa()))
                print("TERMINÓ LA COSA. done = True")
                self.done=True
                return RECOMPENSA
            else:
                print("Bien, satisfaces la demanda: volumen = "+str(self.VOLUMEN_PRODUCIDO)+", demanda = "+str(self.DEMANDA[str(self.t)]))
                print((self.DEMANDA[str(self.t)] / self.VOLUMEN_PRODUCIDO))
                if self.DEMANDA[str(self.t)] / self.VOLUMEN_PRODUCIDO > 0.8:
                  RECOMPENSA=100_000*(self.DEMANDA[str(self.t)] / self.VOLUMEN_PRODUCIDO)
                else:
                  RECOMPENSA+=10_000*(self.DEMANDA[str(self.t)] / self.VOLUMEN_PRODUCIDO)
                #print("Utilidad en periodo "+str(self.t)+": "+str(self.recompensa()))
                self.t+=1
                self.VOLUMEN_PRODUCIDO=0
                self.CORTE_ANTERIOR=[]
                self.CONSTRUIDOS_ANTERIOR=[]
                self.COSTO = 0
                self.INGRESO = 0
                self.total_reward=0
                self.reward=0
                self.prev_reward=0
                return RECOMPENSA
        else: # si no se satisface demanda
            RECOMPENSA+=2_540_321
            print("NO SATISFACES LA DEMANDA. Utilidad en periodo "+str(self.t)+": "+str(self.recompensa()))
            print("volumen = "+str(self.VOLUMEN_PRODUCIDO)+", demanda = "+str(self.DEMANDA[str(self.t)]))
            self.done = True
            print("LO PIERDES TODO - done = True")
            return RECOMPENSA
        
                  
    def reset(self): #función que se activa cuando self.done=True y devuelve todo al estado inicial   
        self.caminos_existentes=self.caminos_existentes_inicial.copy()
        self.caminos_posibles=self.caminos_posibles_inicial.copy()
        self.origenes_existentes=self.origenes_existentes_inicial.copy()
        self.t=2
        self.CORTADOS_TOTALES=[]
        self.CONSTRUIDOS_TOTAL=[]
        self.CORTE_ANTERIOR = []
        self.CORTADOS_TOTALES = []
        self.done=False
        self.INGRESO=0
        self.COSTO=0
        self.VOLUMEN_PRODUCIDO=0
        self.total_reward=0
        self.reward=0
        self.prev_reward=0
        self.state = self.estado()
        return self.state
                             
    def step(self, action):
        print(action)
        if self.is_valid_action(action):
            if action == 44 and self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
                recompensa = self.siguiente_periodo()
                return self.estado(), recompensa, self.done, {}
            else:
                print("ACCION NO VáLIDA")
                self.COSTO += 1e10
                print(self.leer_ambiente())
                print(self.actions_space())
                print(self.available_actions_mask())
                return self.estado(), self.recompensa(), self.done, {}
        #EN CASO DE QUITAR LA MÁSCARA DEL AGENTE, AGREGAR AQUí UN FILTRO CON is.valid_action() QUE PENALICE AL AGENTE, PERO QUE NO LE OBLIGUE A RESETEAR EL ENTORNO MEDIANTE reset()
        else:
            
            print(len(self.CORTADOS_TOTALES))
            print(self.CORTE_ANTERIOR)   
            print(self.CONSTRUIDOS_TOTAL)
            print("rodales disponibles: "+str(self.rodales_existentes()))
            print("Te encuentras en el periodo: "+str(self.t)+", y se elige la acción: "+str(action))

            # Si la acción es cortar un rodal
            if 0 <= action < self.CANTIDAD_RODALES:
                self.cortar_rodal("rodal_"+str(action+1))
                if len(self.CORTADOS_TOTALES) == 12:
                    if self.t < 4:
                        print("No puedes cortar ningun rodal el proximo periodo o sobrepasarías el límite :/")
                        self.COSTO+=2_540_321
                        self.done = True
                        return self.estado(), self.recompensa(), self.done, {}
                    elif self.t == 4:
                        if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
                            print("Al menos satisfaces la demanda :)")
                            recompensa = self.siguiente_periodo()
                            return self.estado(), recompensa, self.done, {}
                        else:
                            self.COSTO+=2_540_321
                            print("Pero no satisface la demanda :(")
                            self.done = True
                            return self.estado(), self.recompensa(), self.done, {} 
                    elif self.t > 4:
                        return self.estado(), self.recompensa(), self.done, {}
                elif len(self.CORTADOS_TOTALES) == 25:
                    print("NO SE PUEDE SEGUIR CORTANDO")
                    if self.t >= 7:
                        if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
                            print("BIEEEN satisfaces la demanda :)")
                            recompensa = self.siguiente_periodo()
                            return self.estado(), recompensa, self.done, {}
                        else:
                            print("ESTÁS EN EL ÚLTIMO PERIODO (7), PERO NO SATISFACES LA DEMANDA :(")
                            recompensa=self.siguiente_periodo()
                            return self.estado(), recompensa, self.done, {}
                    else:
                        self.COSTO+=2_540_321
                        print("No estás en el último periodo y ya no tienes más rodales que cortar... NO satisface la demanda :(")
                        self.done = True
                        return self.estado(), self.recompensa(), self.done, {}
                elif self.rodales_existentes() == [] and (len(self.CORTADOS_TOTALES) < 25 and len(self.CONSTRUIDOS_TOTAL) >= 19):
                    print("##############################Te quedaste sin opciones de corte y ya construiste todos los caminos. Esto debería pasar luego del periodo 4 ###################################")
                    #Esto ocurrirá independiente del periodo de tiempo en que se encuentre el agente
                    if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
                        print("pero SÍ satisfaces la demanda!!! :)")
                        recompensa=self.siguiente_periodo()
                        return self.estado(),recompensa, self.done, {}
                    else:
                        print("Aunque NO satisfaces la demanda :(")
                        recompensa=self.siguiente_periodo()
                        return self.estado(), recompensa, self.done, {}
                elif self.rodales_existentes() != [] and (len(self.CORTADOS_TOTALES) < 25 and len(self.CONSTRUIDOS_TOTAL) >= 19):
                    print("##############################TE QUEDAN OPCIONES, YA SE CONSTRUYERON TODOS LOS CAMINOS ###################################")
                    print("Ambiente: ")
                    print(self.leer_ambiente())
                    return self.estado(), self.recompensa(), self.done, {} 
                
                else:
                    print("CONTINúE")
                    return self.estado(), self.recompensa(), self.done, {}

            # Si la acción es construir un camino
            elif self.CANTIDAD_RODALES <= action < self.CANTIDAD_RODALES + self.CANTIDAD_CAMINOS:
                self.construir_camino("camino_"+str(action - self.CANTIDAD_RODALES+1))
                if self.rodales_existentes() == [] and (len(self.CORTADOS_TOTALES) < 25 and len(self.CONSTRUIDOS_TOTAL) >= 19):
                    print("##############################Te quedaste sin opciones de corte y ya construiste todos los caminos. Esto debería pasar luego del periodo 4 ###################################")
                    #Esto ocurrirá independiente del periodo de tiempo en que se encuentre el agente
                    if self.VOLUMEN_PRODUCIDO >= self.DEMANDA[str(self.t)]:
                        print("pero SÍ satisfaces la demanda!!! :)")
                        recompensa=self.siguiente_periodo()
                        return self.estado(), recompensa, self.done, {}
                    else:
                        print("Aunque NO satisfaces la demanda :(")
                        recompensa=self.siguiente_periodo()
                        return self.estado(), recompensa, self.done, {} 
                else:
                    return self.estado(), self.recompensa(), self.done, {}

            # Si la acción es pasar al siguiente período
            elif action == self.CANTIDAD_RODALES + self.CANTIDAD_CAMINOS:
                recompensa=self.siguiente_periodo()
                return self.estado(), recompensa, self.done, {}
           
                
            

#incluir en vez de la demanda y el volumen por separado, una razon demanda/volumen, acompañado del numero de rodales cortados y el numero de cmainos construidos