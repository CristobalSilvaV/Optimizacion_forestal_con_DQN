# Importamos las librerías necesarias
from pyomo.environ import *
import networkx as nx
import math
import numpy as np
import matplotlib.pyplot as plt

# Diccionario de distancias (i, j) medidas en decenas de kilometros
distancia = {(1, 9): 20, (1, 2): 30, (2, 1): 30, (2, 9): 30, (2, 3): 40, (3, 2): 40, (3, 9): 45, (3, 4): 30, (3, 13): 17, (4, 3): 30, (4, 12): 23, (5, 7): 20, (5, 10): 6, (6, 12): 5, (6, 8): 35, (7, 8): 10, (7, 10): 45, (7, 5): 20, (8, 6): 35, (8, 11): 30, (8, 7): 10, (9, 1): 20, (9, 2): 30, (9, 3): 45, (9, 13): 50, (10, 11): 15, (10, 7): 45, (10, 5): 6, (11, 12): 18, (11, 8): 30, (11, 10): 15, (12, 4): 23, (12, 13): 20, (12, 6): 5, (12, 11): 18, (13, 9): 50, (13, 3): 17, (13, 12): 20}
#Diccionario de precios
p = {1: 92*2, 2: 100*2, 3: 90*2, 4: 93*1.2, 5: 94*1.2, 6: 100*1.2, 7: 95*1.2} #Entre 180 y 200 USD el precio de 1 metro cubico
#Diccionario de costos de costrucción
c_const = {((1, 9), 1): 0.0,
 ((1, 9), 2): 29993.0,
 ((1, 9), 3): 29911.5,
 ((1, 9), 4): 29795.0,
 ((1, 9), 5): 29821.0,
 ((1, 9), 6): 29960.0,
 ((1, 9), 7): 29796.0,
 ((1, 2), 1): 0.0,
 ((1, 2), 2): 29962.0,
 ((1, 2), 3): 29873.5,
 ((1, 2), 4): 29918.5,
 ((1, 2), 5): 29851.0,
 ((1, 2), 6): 29980.5,
 ((1, 2), 7): 29790.5,
 ((2, 1), 1): 0.0,
 ((2, 1), 2): 29805.0,
 ((2, 1), 3): 29873.5,
 ((2, 1), 4): 29867.5,
 ((2, 1), 5): 29774.0,
 ((2, 1), 6): 29960.5,
 ((2, 1), 7): 29886.0,
 ((2, 9), 1): 0.0,
 ((2, 9), 2): 29972.5,
 ((2, 9), 3): 29853.5,
 ((2, 9), 4): 29940.5,
 ((2, 9), 5): 29980.0,
 ((2, 9), 6): 29960.0,
 ((2, 9), 7): 29776.0,
 ((2, 3), 1): 0.0,
 ((2, 3), 2): 29788.5,
 ((2, 3), 3): 29954.0,
 ((2, 3), 4): 29967.5,
 ((2, 3), 5): 29951.0,
 ((2, 3), 6): 29829.0,
 ((2, 3), 7): 29755.0,
 ((3, 2), 1): 0.0,
 ((3, 2), 2): 29902.0,
 ((3, 2), 3): 29959.0,
 ((3, 2), 4): 29796.0,
 ((3, 2), 5): 29926.0,
 ((3, 2), 6): 29859.5,
 ((3, 2), 7): 29759.5,
 ((3, 9), 1): 0.0,
 ((3, 9), 2): 29923.5,
 ((3, 9), 3): 29908.5,
 ((3, 9), 4): 29944.5,
 ((3, 9), 5): 29872.5,
 ((3, 9), 6): 29854.5,
 ((3, 9), 7): 29825.0,
 ((3, 4), 1): 0.0,
 ((3, 4), 2): 29767.0,
 ((3, 4), 3): 29761.0,
 ((3, 4), 4): 29785.5,
 ((3, 4), 5): 29883.0,
 ((3, 4), 6): 29874.0,
 ((3, 4), 7): 29796.5,
 ((3, 13), 1): 0.0,
 ((3, 13), 2): 29889.5,
 ((3, 13), 3): 29926.0,
 ((3, 13), 4): 29796.0,
 ((3, 13), 5): 29786.0,
 ((3, 13), 6): 29804.0,
 ((3, 13), 7): 29789.0,
 ((4, 3), 1): 0.0,
 ((4, 3), 2): 29843.5,
 ((4, 3), 3): 29915.5,
 ((4, 3), 4): 29889.5,
 ((4, 3), 5): 29833.0,
 ((4, 3), 6): 29832.0,
 ((4, 3), 7): 29817.0,
 ((4, 12), 1): 0.0,
 ((4, 12), 2): 29874.5,
 ((4, 12), 3): 29757.5,
 ((4, 12), 4): 29795.0,
 ((4, 12), 5): 29789.0,
 ((4, 12), 6): 29847.5,
 ((4, 12), 7): 29838.5,
 ((5, 7), 1): 0.0,
 ((5, 7), 2): 29936.0,
 ((5, 7), 3): 29769.0,
 ((5, 7), 4): 29761.0,
 ((5, 7), 5): 29786.0,
 ((5, 7), 6): 29974.0,
 ((5, 7), 7): 29951.5,
 ((5, 10), 1): 0.0,
 ((5, 10), 2): 29772.0,
 ((5, 10), 3): 29983.5,
 ((5, 10), 4): 29814.5,
 ((5, 10), 5): 29996.5,
 ((5, 10), 6): 29935.0,
 ((5, 10), 7): 29806.0,
 ((6, 12), 1): 0.0,
 ((6, 12), 2): 29981.0,
 ((6, 12), 3): 29758.5,
 ((6, 12), 4): 29922.5,
 ((6, 12), 5): 29916.0,
 ((6, 12), 6): 29897.5,
 ((6, 12), 7): 29928.0,
 ((6, 8), 1): 0.0,
 ((6, 8), 2): 29903.0,
 ((6, 8), 3): 29778.0,
 ((6, 8), 4): 29898.0,
 ((6, 8), 5): 29769.5,
 ((6, 8), 6): 29926.5,
 ((6, 8), 7): 29904.5,
 ((7, 8), 1): 0.0,
 ((7, 8), 2): 29772.5,
 ((7, 8), 3): 29977.0,
 ((7, 8), 4): 29985.5,
 ((7, 8), 5): 29972.5,
 ((7, 8), 6): 29974.5,
 ((7, 8), 7): 29982.5,
 ((7, 10), 1): 0.0,
 ((7, 10), 2): 29848.0,
 ((7, 10), 3): 29902.0,
 ((7, 10), 4): 29756.5,
 ((7, 10), 5): 29965.5,
 ((7, 10), 6): 29958.0,
 ((7, 10), 7): 29760.5,
 ((7, 5), 1): 0.0,
 ((7, 5), 2): 29828.5,
 ((7, 5), 3): 29782.0,
 ((7, 5), 4): 29862.5,
 ((7, 5), 5): 29783.5,
 ((7, 5), 6): 29794.5,
 ((7, 5), 7): 29887.5,
 ((8, 6), 1): 0.0,
 ((8, 6), 2): 29939.5,
 ((8, 6), 3): 29779.5,
 ((8, 6), 4): 29914.0,
 ((8, 6), 5): 29978.0,
 ((8, 6), 6): 29776.0,
 ((8, 6), 7): 29803.5,
 ((8, 11), 1): 0.0,
 ((8, 11), 2): 29766.0,
 ((8, 11), 3): 29852.0,
 ((8, 11), 4): 29763.0,
 ((8, 11), 5): 29937.0,
 ((8, 11), 6): 29816.0,
 ((8, 11), 7): 29802.5,
 ((8, 7), 1): 0.0,
 ((8, 7), 2): 29832.0,
 ((8, 7), 3): 29942.5,
 ((8, 7), 4): 29846.0,
 ((8, 7), 5): 29939.5,
 ((8, 7), 6): 29751.0,
 ((8, 7), 7): 29788.0,
 ((9, 1), 1): 0.0,
 ((9, 1), 2): 29802.5,
 ((9, 1), 3): 29916.5,
 ((9, 1), 4): 29811.5,
 ((9, 1), 5): 29846.5,
 ((9, 1), 6): 29901.5,
 ((9, 1), 7): 29761.5,
 ((9, 2), 1): 0.0,
 ((9, 2), 2): 29994.0,
 ((9, 2), 3): 29815.5,
 ((9, 2), 4): 29916.0,
 ((9, 2), 5): 29974.0,
 ((9, 2), 6): 29988.5,
 ((9, 2), 7): 29944.5,
 ((9, 3), 1): 0.0,
 ((9, 3), 2): 29972.5,
 ((9, 3), 3): 29796.5,
 ((9, 3), 4): 29987.5,
 ((9, 3), 5): 29851.0,
 ((9, 3), 6): 29998.0,
 ((9, 3), 7): 29832.0,
 ((9, 13), 1): 0.0,
 ((9, 13), 2): 29781.5,
 ((9, 13), 3): 29763.5,
 ((9, 13), 4): 29928.5,
 ((9, 13), 5): 29990.5,
 ((9, 13), 6): 29782.5,
 ((9, 13), 7): 29924.5,
 ((10, 11), 1): 0.0,
 ((10, 11), 2): 29807.5,
 ((10, 11), 3): 29911.5,
 ((10, 11), 4): 29917.0,
 ((10, 11), 5): 29936.0,
 ((10, 11), 6): 29755.5,
 ((10, 11), 7): 29825.0,
 ((10, 7), 1): 0.0,
 ((10, 7), 2): 29934.0,
 ((10, 7), 3): 29976.5,
 ((10, 7), 4): 29910.5,
 ((10, 7), 5): 29876.5,
 ((10, 7), 6): 29850.0,
 ((10, 7), 7): 29795.5,
 ((10, 5), 1): 0.0,
 ((10, 5), 2): 29988.5,
 ((10, 5), 3): 29860.5,
 ((10, 5), 4): 29824.5,
 ((10, 5), 5): 29832.0,
 ((10, 5), 6): 29771.0,
 ((10, 5), 7): 29980.5,
 ((11, 12), 1): 0.0,
 ((11, 12), 2): 29837.5,
 ((11, 12), 3): 29894.0,
 ((11, 12), 4): 29848.0,
 ((11, 12), 5): 29876.0,
 ((11, 12), 6): 29751.5,
 ((11, 12), 7): 29888.5,
 ((11, 8), 1): 0.0,
 ((11, 8), 2): 29796.5,
 ((11, 8), 3): 29844.5,
 ((11, 8), 4): 29856.5,
 ((11, 8), 5): 29816.0,
 ((11, 8), 6): 29919.5,
 ((11, 8), 7): 29958.5,
 ((11, 10), 1): 0.0,
 ((11, 10), 2): 29910.0,
 ((11, 10), 3): 29805.0,
 ((11, 10), 4): 29882.5,
 ((11, 10), 5): 29954.0,
 ((11, 10), 6): 29926.5,
 ((11, 10), 7): 29971.5,
 ((12, 4), 1): 0.0,
 ((12, 4), 2): 29877.5,
 ((12, 4), 3): 29994.0,
 ((12, 4), 4): 29782.0,
 ((12, 4), 5): 29956.0,
 ((12, 4), 6): 29809.0,
 ((12, 4), 7): 29987.5,
 ((12, 13), 1): 0.0,
 ((12, 13), 2): 29939.5,
 ((12, 13), 3): 29808.5,
 ((12, 13), 4): 29835.5,
 ((12, 13), 5): 29928.0,
 ((12, 13), 6): 29946.5,
 ((12, 13), 7): 29807.5,
 ((12, 6), 1): 0.0,
 ((12, 6), 2): 29753.0,
 ((12, 6), 3): 29902.0,
 ((12, 6), 4): 29898.5,
 ((12, 6), 5): 29795.0,
 ((12, 6), 6): 29889.0,
 ((12, 6), 7): 29776.5,
 ((12, 11), 1): 0.0,
 ((12, 11), 2): 29926.0,
 ((12, 11), 3): 29873.0,
 ((12, 11), 4): 29849.5,
 ((12, 11), 5): 29848.0,
 ((12, 11), 6): 29962.0,
 ((12, 11), 7): 29877.0,
 ((13, 9), 1): 0.0,
 ((13, 9), 2): 29897.5,
 ((13, 9), 3): 29878.5,
 ((13, 9), 4): 29847.5,
 ((13, 9), 5): 29877.0,
 ((13, 9), 6): 29753.0,
 ((13, 9), 7): 29948.0,
 ((13, 3), 1): 0.0,
 ((13, 3), 2): 29964.5,
 ((13, 3), 3): 29818.0,
 ((13, 3), 4): 29781.0,
 ((13, 3), 5): 29814.0,
 ((13, 3), 6): 29938.5,
 ((13, 3), 7): 29864.0,
 ((13, 12), 1): 0.0,
 ((13, 12), 2): 29864.0,
 ((13, 12), 3): 29997.0,
 ((13, 12), 4): 29750.5,
 ((13, 12), 5): 29755.5,
 ((13, 12), 6): 29831.0,
 ((13, 12), 7): 29990.5} #60.000 USD por construir un camino en promedio
#Diccionario de costos de corte
c_corte = {(1, 1): 5466.5,
 (1, 2): 5834.0,
 (1, 3): 5331.0,
 (1, 4): 5615.5,
 (1, 5): 5653.0,
 (1, 6): 5337.5,
 (1, 7): 5758.5,
 (2, 1): 5916.5,
 (2, 2): 5173.0,
 (2, 3): 5635.0,
 (2, 4): 5006.5,
 (2, 5): 5479.5,
 (2, 6): 5324.0,
 (2, 7): 5690.0,
 (3, 1): 5493.0,
 (3, 2): 5905.5,
 (3, 3): 5648.5,
 (3, 4): 5371.0,
 (3, 5): 5656.0,
 (3, 6): 5023.5,
 (3, 7): 5184.0,
 (4, 1): 5961.5,
 (4, 2): 5714.5,
 (4, 3): 5954.5,
 (4, 4): 5914.5,
 (4, 5): 5649.5,
 (4, 6): 5008.5,
 (4, 7): 5233.5,
 (5, 1): 5379.0,
 (5, 2): 5303.0,
 (5, 3): 5064.5,
 (5, 4): 5301.5,
 (5, 5): 5749.5,
 (5, 6): 5956.0,
 (5, 7): 5595.0,
 (6, 1): 5064.5,
 (6, 2): 5203.5,
 (6, 3): 5293.5,
 (6, 4): 5131.0,
 (6, 5): 5559.5,
 (6, 6): 5915.0,
 (6, 7): 5427.0,
 (7, 1): 5585.0,
 (7, 2): 5147.0,
 (7, 3): 5249.5,
 (7, 4): 5652.5,
 (7, 5): 5392.5,
 (7, 6): 5952.0,
 (7, 7): 5485.0,
 (8, 1): 5497.5,
 (8, 2): 5534.5,
 (8, 3): 5371.0,
 (8, 4): 5223.0,
 (8, 5): 5074.5,
 (8, 6): 5996.0,
 (8, 7): 5809.5,
 (9, 1): 5449.0,
 (9, 2): 5391.0,
 (9, 3): 5637.5,
 (9, 4): 5735.0,
 (9, 5): 5291.0,
 (9, 6): 5836.5,
 (9, 7): 5002.0,
 (10, 1): 5494.5,
 (10, 2): 5574.0,
 (10, 3): 5984.0,
 (10, 4): 5962.0,
 (10, 5): 5127.0,
 (10, 6): 5646.5,
 (10, 7): 5120.5,
 (11, 1): 5606.5,
 (11, 2): 5273.0,
 (11, 3): 5738.0,
 (11, 4): 5758.0,
 (11, 5): 5531.5,
 (11, 6): 5911.5,
 (11, 7): 5317.0,
 (12, 1): 5167.5,
 (12, 2): 5767.0,
 (12, 3): 5026.5,
 (12, 4): 5731.5,
 (12, 5): 5090.0,
 (12, 6): 5992.5,
 (12, 7): 5051.0,
 (13, 1): 5341.5,
 (13, 2): 5253.0,
 (13, 3): 5352.0,
 (13, 4): 5668.5,
 (13, 5): 5927.0,
 (13, 6): 5863.0,
 (13, 7): 5358.0,
 (14, 1): 5363.0,
 (14, 2): 5806.0,
 (14, 3): 5580.5,
 (14, 4): 5210.5,
 (14, 5): 5983.5,
 (14, 6): 5834.0,
 (14, 7): 5318.0,
 (15, 1): 5162.0,
 (15, 2): 5072.0,
 (15, 3): 5542.0,
 (15, 4): 5207.0,
 (15, 5): 5263.5,
 (15, 6): 5964.5,
 (15, 7): 5280.5,
 (16, 1): 5115.0,
 (16, 2): 5702.5,
 (16, 3): 5970.5,
 (16, 4): 5838.5,
 (16, 5): 5881.0,
 (16, 6): 5644.0,
 (16, 7): 5967.5,
 (17, 1): 5868.5,
 (17, 2): 5994.5,
 (17, 3): 5091.5,
 (17, 4): 5475.5,
 (17, 5): 5045.5,
 (17, 6): 5507.0,
 (17, 7): 5048.5,
 (18, 1): 5381.0,
 (18, 2): 5615.5,
 (18, 3): 5759.0,
 (18, 4): 5521.5,
 (18, 5): 5591.0,
 (18, 6): 5284.0,
 (18, 7): 5776.5,
 (19, 1): 5401.0,
 (19, 2): 5446.5,
 (19, 3): 5232.5,
 (19, 4): 5654.0,
 (19, 5): 5220.5,
 (19, 6): 5145.0,
 (19, 7): 6000.0,
 (20, 1): 5614.5,
 (20, 2): 5468.0,
 (20, 3): 5093.5,
 (20, 4): 5649.0,
 (20, 5): 5016.5,
 (20, 6): 5531.0,
 (20, 7): 5221.0,
 (21, 1): 5341.5,
 (21, 2): 5738.0,
 (21, 3): 5078.5,
 (21, 4): 5592.5,
 (21, 5): 5228.5,
 (21, 6): 5000.5,
 (21, 7): 5878.0,
 (22, 1): 5587.5,
 (22, 2): 5767.0,
 (22, 3): 5723.0,
 (22, 4): 5545.0,
 (22, 5): 5107.0,
 (22, 6): 5400.5,
 (22, 7): 5919.0,
 (23, 1): 5986.5,
 (23, 2): 5885.5,
 (23, 3): 5663.5,
 (23, 4): 5661.0,
 (23, 5): 5750.0,
 (23, 6): 5801.0,
 (23, 7): 5379.0,
 (24, 1): 5153.5,
 (24, 2): 5709.0,
 (24, 3): 5405.0,
 (24, 4): 5579.0,
 (24, 5): 5960.5,
 (24, 6): 5724.5,
 (24, 7): 5044.0,
 (25, 1): 5805.5,
 (25, 2): 5339.5,
 (25, 3): 5283.5,
 (25, 4): 5667.5,
 (25, 5): 5247.0,
 (25, 6): 5414.0,
 (25, 7): 5460.0} #20 USD x un promedio de 600 decenas de metros cubicos
#Diccionario de produccion de volumen
P = {(1, 1): 830,
 (1, 2): 839,
 (1, 3): 651,
 (1, 4): 526,
 (1, 5): 914,
 (1, 6): 764,
 (1, 7): 662,
 (2, 1): 571,
 (2, 2): 553,
 (2, 3): 548,
 (2, 4): 935,
 (2, 5): 516,
 (2, 6): 537,
 (2, 7): 540,
 (3, 1): 731,
 (3, 2): 912,
 (3, 3): 556,
 (3, 4): 691,
 (3, 5): 522,
 (3, 6): 860,
 (3, 7): 521,
 (4, 1): 512,
 (4, 2): 758,
 (4, 3): 849,
 (4, 4): 928,
 (4, 5): 771,
 (4, 6): 918,
 (4, 7): 816,
 (5, 1): 832,
 (5, 2): 708,
 (5, 3): 736,
 (5, 4): 708,
 (5, 5): 712,
 (5, 6): 737,
 (5, 7): 826,
 (6, 1): 705,
 (6, 2): 752,
 (6, 3): 577,
 (6, 4): 946,
 (6, 5): 639,
 (6, 6): 604,
 (6, 7): 671,
 (7, 1): 610,
 (7, 2): 665,
 (7, 3): 644,
 (7, 4): 582,
 (7, 5): 744,
 (7, 6): 635,
 (7, 7): 829,
 (8, 1): 817,
 (8, 2): 894,
 (8, 3): 881,
 (8, 4): 686,
 (8, 5): 618,
 (8, 6): 934,
 (8, 7): 943,
 (9, 1): 510,
 (9, 2): 624,
 (9, 3): 818,
 (9, 4): 555,
 (9, 5): 587,
 (9, 6): 733,
 (9, 7): 945,
 (10, 1): 527,
 (10, 2): 627,
 (10, 3): 687,
 (10, 4): 874,
 (10, 5): 745,
 (10, 6): 616,
 (10, 7): 618,
 (11, 1): 823,
 (11, 2): 630,
 (11, 3): 527,
 (11, 4): 855,
 (11, 5): 878,
 (11, 6): 614,
 (11, 7): 920,
 (12, 1): 572,
 (12, 2): 622,
 (12, 3): 807,
 (12, 4): 517,
 (12, 5): 838,
 (12, 6): 521,
 (12, 7): 787,
 (13, 1): 817,
 (13, 2): 674,
 (13, 3): 814,
 (13, 4): 587,
 (13, 5): 774,
 (13, 6): 639,
 (13, 7): 626,
 (14, 1): 567,
 (14, 2): 660,
 (14, 3): 860,
 (14, 4): 693,
 (14, 5): 555,
 (14, 6): 533,
 (14, 7): 559,
 (15, 1): 944,
 (15, 2): 895,
 (15, 3): 571,
 (15, 4): 635,
 (15, 5): 573,
 (15, 6): 743,
 (15, 7): 938,
 (16, 1): 522,
 (16, 2): 718,
 (16, 3): 813,
 (16, 4): 746,
 (16, 5): 688,
 (16, 6): 886,
 (16, 7): 804,
 (17, 1): 679,
 (17, 2): 585,
 (17, 3): 897,
 (17, 4): 589,
 (17, 5): 516,
 (17, 6): 680,
 (17, 7): 927,
 (18, 1): 948,
 (18, 2): 899,
 (18, 3): 667,
 (18, 4): 529,
 (18, 5): 649,
 (18, 6): 750,
 (18, 7): 657,
 (19, 1): 654,
 (19, 2): 534,
 (19, 3): 677,
 (19, 4): 627,
 (19, 5): 800,
 (19, 6): 547,
 (19, 7): 887,
 (20, 1): 941,
 (20, 2): 896,
 (20, 3): 722,
 (20, 4): 861,
 (20, 5): 676,
 (20, 6): 886,
 (20, 7): 877,
 (21, 1): 828,
 (21, 2): 529,
 (21, 3): 535,
 (21, 4): 646,
 (21, 5): 664,
 (21, 6): 723,
 (21, 7): 637,
 (22, 1): 661,
 (22, 2): 570,
 (22, 3): 780,
 (22, 4): 899,
 (22, 5): 662,
 (22, 6): 848,
 (22, 7): 943,
 (23, 1): 565,
 (23, 2): 531,
 (23, 3): 566,
 (23, 4): 900,
 (23, 5): 742,
 (23, 6): 538,
 (23, 7): 781,
 (24, 1): 769,
 (24, 2): 801,
 (24, 3): 658,
 (24, 4): 765,
 (24, 5): 780,
 (24, 6): 652,
 (24, 7): 759,
 (25, 1): 927,
 (25, 2): 877,
 (25, 3): 800,
 (25, 4): 592,
 (25, 5): 648,
 (25, 6): 877,
 (25, 7): 525} #decenas de metros cubicos
#Diccionario de demanda
D = {1: 0, 2: 2608/2, 3: 2638/2, 4: 2692/2, 5: 2678/2, 6: 2632/2, 7: 2615/2}
# Creación del modelo
model = ConcreteModel()

# Conjuntos

caminos_iniciales = [(2, 3), (3, 2), (3, 13), (13, 3), (5, 10), (10, 5), (10, 11), (11, 10), (12, 11), (11, 12), (12, 13), (13, 12)]
model.CaminosIniciales = Set(initialize=caminos_iniciales)
RA = {1: [1,2,4],2: [6,7,9,10], 3: [12,13], 4: [11,15,16], 5: [14,23,24], 6: [17,18,19], 7: [22,25], 8: [20,21], 9: [3,5,8]}
RAR = {1: [2,4],2:[1,3,5],3:[2,5,13],4:[1,5,6,9],5:[2,3,4,6,8],6:[4,5,7,8,9],7:[6,8,10,12],8:[6,7,5,12,13],9:[4,6,10],10:[9,7,11],11:[10,12,15,16],12:[7,8,11,13,16,17],13:[3,8,12,17],14:[11,15,20,23,24],15:[11,16,20,14],16:[11,12,15,17,19],17:[12,13,16,18],18:[17,19],19:[18,16,20,21],20:[14,15,19,23],21:[19,20,22,23,25],22:[21,25],23:[20,21,25,24,14],24:[14,23],25:[21,22,23]}
N_predecesor = {2:[1,9], 3: [2,4,9], 5: [7], 6: [8], 8: [7], 9: [1,2], 10: [5,7], 11: [8,10], 12: [4,6,11], 13: [3,9,12]}
N_sucesor = {1:[2,9], 2:[3,9], 3: [9,13], 4: [3,12], 5: [10], 6: [12], 7: [5,8,10], 8: [6,11], 9: [2,3,13], 10: [11], 11: [12], 12: [13]}
model.RAR = Set(initialize = [(r,ra) for r in RAR for ra in RAR[r]] ,doc = 'Pares (rodal,rodal_asociado)')
model.RA = Set(initialize=[(r, n) for r in RA for n in RA[r]], doc='Pares (rodal, nodo)')
model.N_predecesor = Set(initialize=[(a, b) for a in N_predecesor for b in N_predecesor[a]], doc='Pares (nodo, predecesor al nodo)')
model.N_sucesor = Set(initialize=[(a, b) for a in N_sucesor for b in N_sucesor[a]], doc='Pares (nodo, sucesor al nodo)')

model.T = Set(initialize=range(1, 8), doc='Periodos de tiempo')
model.R = Set(initialize=range(1, 26), doc='Rodales')
model.N = Set(initialize=range(1, 14), doc='Nodos')
model.A = Set(initialize=distancia.keys(), doc='Aristas')

model.N_origen = Set(initialize=[1, 2, 3, 4, 5, 6, 7, 8, 9], doc='Nodos de origen')
model.N_interseccion = Set(initialize=[10, 11, 12], doc='Nodos de intersección')
model.N_salida = Set(initialize=[13], doc='Nodo de salida')

# Parámetros
model.d = Param(model.A, initialize = distancia)
model.p = Param(model.T, initialize = p, doc='Precio de venta por metro cúbico en cada periodo')
model.k = Param(initialize = 12.02, doc='Costo fijo por transportar un metro cúbico por kilómetro') #costo medido en USD
model.c_const = Param(model.A, model.T, initialize = c_const, doc='Costo de construcción de cada arista/camino (i,j) en el periodo t')
model.c_corte = Param(model.R, model.T, initialize = c_corte, doc='Costo de corte de cada rodal en el periodo t')
model.P = Param(model.R, model.T, initialize = P, doc='Producción del rodal r en el periodo t')
model.D = Param(model.T, initialize = D, doc='Demanda por volumen de madera en cada periodo')

# Variables
model.E = Var(model.R, model.T, within=Binary, initialize = 0, doc='Si se corta o no el rodal r en el periodo t')
model.X = Var(model.A, model.T, within=Binary, initialize = 0, doc='Si el camino que une los nodos i,j está disponible o no en el periodo t')
model.W = Var(model.A, model.T, within=Binary, initialize = 0, doc='Si se construye o no el camino que une los nodos i,j en el periodo t')
model.Y = Var(model.N_origen, model.T, within=NonNegativeReals, initialize = 0, doc='Volumen enviado desde cada nodo origen en el periodo t')
model.F = Var(model.A, model.T, within=NonNegativeReals, initialize = 0, doc='Flujo de volumen enviado entre nodos en el periodo t')
model.COSTO_TRANSPORTE = Var(model.T,within=NonNegativeReals, initialize = 0)
model.COSTO_CONSTRUCCION = Var(model.T,within=NonNegativeReals, initialize = 0)
model.COSTO_COSECHA = Var(model.T,within=NonNegativeReals, initialize = 0)
model.INGRESO = Var(model.T,within=NonNegativeReals, initialize = 0)



# Declara un diccionario para almacenar los valores iniciales
# Función objetivo
def objective_rule(model):
    return sum(model.INGRESO[t] - model.COSTO_TRANSPORTE[t] - model.COSTO_CONSTRUCCION[t] - model.COSTO_COSECHA[t] for t in model.T)
model.objective = Objective(rule=objective_rule, sense=maximize, doc='Maximizar las utilidades del administrador forestal en cada periodo de tiempo t')

# Restricciones


def demand_rule(model, t):
    return sum(model.E[r, t] * model.P[r, t] for r in model.R) >= model.D[t]
model.demand = Constraint(model.T, rule=demand_rule, doc='La demanda debe ser satisfecha con el volumen generado')

'''

CAMINOS

'''
def costo_transporte(model,t):
    return sum(model.k * model.d[i, j] * model.F[i, j, t] for i,j in model.A) == model.COSTO_TRANSPORTE[t]
model.cost_trans = Constraint(model.T, rule = costo_transporte)

def costo_construccion(model,t):
    return sum(model.c_const[i, j, t]* model.W[(i, j), t]  for i,j in model.A)/2 == model.COSTO_CONSTRUCCION[t] #se divide en 2 porque se considera el costo de solo 1 camino, pero el modelo suma ambos sentidos del camino
model.cost_const = Constraint(model.T, rule = costo_construccion)

def costo_cosecha(model,t):
    return sum(model.c_corte[r, t] * model.E[r, t] for r in model.R) == model.COSTO_COSECHA[t]
model.cost_cosecha = Constraint(model.T, rule = costo_cosecha)

def ingreso_cosecha(model,t):
    return sum(model.p[t] * model.Y[n, t] for n in model.N_origen) == model.INGRESO[t]
model.ingreso_cosecha = Constraint(model.T, rule = ingreso_cosecha)

def get_successors(node, successor_set):
    return [b for a, b in successor_set if a == node]

def get_predecessors(node, predecessor_set):
    return [b for a, b in predecessor_set if a == node]

def get_asociados(rodal,asociados_set):
    return [B for A,B in asociados_set if A==rodal]


def caminos_construidos_unicamente_una_vez_rule(model, i, j):
    return sum(model.W[i, j, t] for t in model.T) <= 1
model.CaminosConstruidosUnicamenteUnaVez = Constraint(model.A, rule=caminos_construidos_unicamente_una_vez_rule)

def W_iguales(model,i,j,t):
    return model.W[i,j,t] == model.W[j,i,t]
model.W_iguales_constraint = Constraint(model.A,model.T, rule = W_iguales, doc ='Los caminos i,j y j,i son el mismo camino')

def X_iguales(model,i,j,t):
    return model.X[i,j,t] == model.X[j,i,t]
model.X_iguales_constraint = Constraint(model.A,model.T, rule = X_iguales, doc ='Los caminos i,j y j,i son el mismo camino')

def no_isolated_paths_rule(model, i, j, t):
    if t == 1:
        if (i, j) in caminos_iniciales:
            return model.W[i, j, t] == 1
        else:
            return model.W[i, j, t] == 0
    else:
        carretera = caminos_iniciales
        for camino in model.A:
                if model.X[camino,t].value == 1:
                    carretera.append(camino)
        return model.W[i, j, t] <= sum(model.X[p, q, t - 1] for p, q in carretera if (p == i or p == j or q == i or q == j) and (p, q) != (i, j))

model.no_isolated_paths = Constraint(model.A, model.T, rule=no_isolated_paths_rule)


def restriccion_transicion(model, i, j, t):
        return model.X[i, j, t] <= sum(model.W[i, j, t_prime] for t_prime in range(1, t+1))
model.restriccion_transicion_X = Constraint(model.A, model.T, rule=restriccion_transicion)


# Restricción para mantener los caminos disponibles una vez construidos
def restriccion_mantener_caminos_disponibles(model, i, j, t):
    if t > 1:
        return model.X[i, j, t] >= model.X[i, j, t - 1]+model.W[i,j,t]
    else:
        return Constraint.Skip
model.restriccion_mantener_caminos_disponibles = Constraint(model.A, model.T, rule=restriccion_mantener_caminos_disponibles)



'''
FLUJOS
'''
def maximo_3(model,t):
    return sum(model.E[r,t] for r in model.R) <= 5
model.harvest_limit_constraint = Constraint(model.T, rule = maximo_3, doc='Se puede cortar máximo 3 rodales por periodo de tiempo')

def flow_rule(model,i,j,t):
    return model.F[i, j, t] <= 1e6 * sum(model.W[i, j, t_prime] for t_prime in range(1, t+1))

model.flow_constraint = Constraint(model.A,model.T, rule = flow_rule, doc='La cantidad de madera que fluye entre dos nodos i y j en un periodo t está limitada por la capacidad del camino entre ellos. Si no hay camino, el flujo debe ser cero')

def intersection_flow_rule(model, n,t):
    return sum(model.F[n,j,t] for j in get_successors(n,model.N_sucesor)) == sum(model.F[i,n,t] for i in get_predecessors(n,model.N_predecesor))
model.intersection_flow_constraint = Constraint(model.N_interseccion,model.T, rule = intersection_flow_rule, doc='En los nodos de intersecci ́on, la cantidad de madera que ingresa debe ser igual a la cantidad de madera que sale en cada periodo t')

def origen_flow_rule(model, n,t):
    return sum(model.F[n,j,t] for j in get_successors(n,model.N_sucesor)) == model.Y[n,t] + sum(model.F[i,n,t] for i in get_predecessors(n,model.N_predecesor))
model.origen_constraint = Constraint(model.N_origen,model.T, rule = origen_flow_rule, doc='En los nodos de origen, la cantidad de madera enviada debe ser igual a la producción de los rodales asociados en cada periodo t')

def harvested_volume_rule(model, m, t):
    return model.Y[m, t] == sum(model.E[r, t] * model.P[r, t] for r in get_asociados(m,model.RA))
model.harvested_volume = Constraint(model.N_origen, model.T, rule=harvested_volume_rule, doc='El volumen enviado desde el nodo origen corresponde a la suma de la producción producto de la cosecha de los rodales')

def flow_to_exit_node_rule(model, t):
    exit_node = model.N_salida.first()
    return sum(model.F[n, exit_node, t] for n in get_predecessors(exit_node, model.N_predecesor)) == sum(model.E[r, t] * model.P[r, t] for m in model.N_origen for r in get_asociados(m, model.RA))

model.flow_to_exit_node = Constraint(model.T, rule=flow_to_exit_node_rule, doc='Flujo hacia el nodo de salida')

def no_flow_from_exit_node_rule(model, t):
    exit_node = model.N_salida.first()
    return sum(model.F[exit_node, n, t] for n in get_predecessors(exit_node, model.N_predecesor)) == 0

model.no_flow_from_exit_node = Constraint(model.T, rule=no_flow_from_exit_node_rule, doc='No debe haber flujo desde el nodo de salida a sus antecesores')


def rodales_cortados_unicamente_una_vez_rule(model, r):
    return sum(model.E[r, t] for t in model.T) <= 1

model.RodalesCortadosUnicamenteUnaVez = Constraint(model.R, rule=rodales_cortados_unicamente_una_vez_rule)

def no_adjacent_harvest_rule(model, r, ra, t):
    return model.E[r, t] + model.E[ra, t] <= 1

model.no_adjacent_harvest = Constraint(model.RAR, model.T, rule=no_adjacent_harvest_rule, doc='No se pueden cortar rodales adyacentes en el mismo periodo')

def periodo_inicial(model):
    return sum(model.E[r,1] for r in model.R) + sum(model.W[i,j,1] + model.X[i,j,1] for i,j in model.A) == 24

model.initial_phase = Constraint(rule=periodo_inicial, doc='Establece que la suma de decisiones del periodo inicial deja como resultado los caminos iniciales disponibles')

print('Final')

def solve_model(model):
    # Solver GLPK
    solver = SolverFactory('gurobi')
    solver.options['mipgap'] = 0.001  # Brecha de tolerancia del 1%
    results = solver.solve(model, tee=True)
    return results

results = solve_model(model)

'''
---------------------------------------------------
---------------------------------------------------
----------RESULTADOS-------------------------------
---------------------------------------------------
'''

data_E = []
for r in model.R:
    for t in model.T:
        if model.E[r,t].value > 0.5:
            data_E.append((r,t,model.E[r,t].value))

data_Y = []
for n in model.N_origen:
   for t in model.T:
        if model.Y[n,t].value > 0.5:
            data_Y.append((n,t,model.Y[n,t].value))

data_W = []
for i,j in model.A:
    for t in model.T:
        if model.W[i,j,t].value > 0.5:
            data_W.append(((i,j),t,model.W[(i,j),t].value))

data_X = []
for i,j in model.A:
    for t in model.T:
        if model.X[i,j,t].value > 0.5:
            data_X.append(((i,j),t,model.X[i,j,t].value))

data_F = []
for i,j in model.A:
    for t in model.T:
        if model.F[i,j,t].value > 0.5:
            data_F.append(((i,j),t,model.F[i,j,t].value))

data_costo_transporte = []
for t in model.T:
    if model.COSTO_TRANSPORTE[t].value > 0.5:
        data_costo_transporte.append((t,model.COSTO_TRANSPORTE[t].value))

data_costo_cosecha = []
for t in model.T:
    if model.COSTO_COSECHA[t].value > 0.5:
        data_costo_cosecha.append((t,model.COSTO_COSECHA[t].value))

data_costo_construccion = []
for t in model.T:
    if model.COSTO_CONSTRUCCION[t].value > 0.5:
        data_costo_construccion.append((t,model.COSTO_CONSTRUCCION[t].value))

data_ingresos = []
for t in model.T:
    if model.INGRESO[t].value > 0.5:
        data_ingresos.append((t,model.INGRESO[t].value))

import pandas as pd

# Crear DataFrames
df_E = pd.DataFrame(data_E, columns=["r", "t", "value"])
df_Y = pd.DataFrame(data_Y, columns=["n", "t", "value"])
df_W = pd.DataFrame(data_W, columns=["pair", "period", "value"])
df_X = pd.DataFrame(data_X, columns=["pair", "period", "value"])
df_F = pd.DataFrame(data_F, columns=["pair", "period", "value"])

# Crear una matriz vacía para los valores de E (25 rodales x 7 periodos)
matrix_E = np.zeros((25, 7))

# Rellenar la matriz con los valores de E en el DataFrame
for _, row in df_E.iterrows():
    rodal = int(row['r']) - 1
    periodo = int(row['t']) - 1
    value = row['value']
    
    matrix_E[rodal, periodo] = value


# Crear un gráfico de barras
fig, ax = plt.subplots()

for i in range(25):
    for j in range(7):
        if matrix_E[i, j] != 0:
            ax.barh(i, 1, left=j, color='purple')
        else:
            ax.barh(i, 1, left=j, color='green')

# Configurar ejes y etiquetas
ax.set_xlabel('Periodos')
ax.set_ylabel('Rodales')
ax.set_xticks(np.arange(0.5, 7.5, 1))
ax.set_xticklabels(range(1, 8))
ax.set_yticks(np.arange(0.5, 25.5, 1))
ax.set_yticklabels(range(1,26))
ax.set_xlim(0, 7)
ax.set_ylim(0, 25)

# Mostrar el gráfico
plt.show()

'''

'''
def calculate_positions_around_node(node_pos, num_positions, radius=0.04):
    angles = np.linspace(0, 2* np.pi, num_positions + 1)[:-1]
    x, y = node_pos
    positions = [(x + radius * np.cos(angle), y + radius * np.sin(angle)) for angle in angles]
    return positions
'''

'''

# Crea el grafo
G = nx.DiGraph()

# Añade los nodos y las aristas al grafo
for edge, weight in distancia.items():
    G.add_edge(edge[0], edge[1], weight=weight)

# Función para dibujar el grafo en cada periodo
def draw_graph_for_period(period):
    plt.figure(figsize=(12, 10))
    pos = {1: np.array([-0.63091702,  0.6]), 9: np.array([-0.5,  0.75866375]), 2: np.array([-0.5,  0]), 3: np.array([-0.35,  0.52]), 4: np.array([-0.3,  0.01]), 13: np.array([-0.26 , 0.83]), 12: np.array([-0.2, 0.4]), 5: np.array([-0.01, -0.3]), 7: np.array([ 0.02152219, 0]), 10: np.array([-0.13, -0.2]), 6: np.array([-0.11, 0.62]), 8: np.array([ -0.01, 0.24]), 11: np.array([ -0.18, 0.1])}

    #print(pos)
    
    # Dibuja las aristas en azul si están en df_X para el periodo dado
    blue_edges = [(row['pair']) for _, row in df_X[df_X['period'] == period].iterrows()]
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='blue', width=2)

    # Dibuja las aristas en amarillo si están en df_W para el periodo dado
    yellow_edges = [(row['pair']) for _, row in df_W[df_W['period'] == period].iterrows()]
    nx.draw_networkx_edges(G, pos, edgelist=yellow_edges, edge_color='yellow', width=3)
    
    # Dibuja las aristas en rojo si están en df_F para el periodo dado
    red_edges = [(row['pair']) for _, row in df_F[df_F['period'] == period].iterrows()]
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', width=2)

      # Añade la cantidad de flujo en las flechas 
    edge_labels = {(row['pair']): math.ceil(row['value']) for _, row in df_F[df_F['period'] == period].iterrows()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_family="sans-serif", font_color='blue')

    # Dibuja los nodos y las etiquetas de los nodos
    nx.draw_networkx_nodes(G, pos, node_size=400)
    nx.draw_networkx_labels(G, pos, font_size=11, font_family="sans-serif")

    # Dibuja círculos alrededor de los nodos con rodales asociados
    for node, rodales in RA.items():
        rodal_positions = calculate_positions_around_node(pos[node], len(rodales))
        for idx, rodal in enumerate(rodales):
            rodal_color = 'red' if any((df_E['r'] == rodal) & (df_E['t'] == period)) else 'green'
            plt.scatter(rodal_positions[idx][0], rodal_positions[idx][1], c=rodal_color, s=250)
            plt.text(rodal_positions[idx][0], rodal_positions[idx][1], str(rodal), fontsize=9, ha='center', va='center')
   


    plt.title(f"Grafo para el periodo {period}, con D[{period}]= {D[period]} y k = {model.k.value}")
    plt.axis("off")
    plt.show()

# Dibuja el grafo para cada periodo
unique_periods = df_X['period'].unique()
for period in unique_periods:
    draw_graph_for_period(period)

valor_optimo = '{:,.0f}'.format(int(model.objective.expr()))
print(f"El valor óptimo de la utilidad es: ${valor_optimo} USD")