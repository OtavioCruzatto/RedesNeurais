from Perceptron import Perceptron
import matplotlib.pyplot as plt


def dados_2():

    x_treino = [[-9.0, -4.0], [-8.0, -3.0], [-7.0, -2.0], [-6.0, -1.0], [-5.0, 0.0],
                [-4.0, 1.0], [-3.0, 2.0], [-2.0, 3.0], [-1.0, 4.0], [0.0, 5.0],
                [1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, 9.0], [5.0, 10.0],
                [6.0, 11.0], [7.0, 12.0], [8.0, 13.0], [9.0, 14.0], [-9.0, -14.0],
                [-8.0, -13.0], [-7.0, -12.0], [-6.0, -11.0], [-5.0, -10.0], [-4.0, -9.0],
                [-3.0, -8.0], [-2.0, -7.0], [-1.0, -6.0], [0.0, -5.0], [1.0, -4.0],
                [2.0, -3.0], [3.0, -2.0], [4.0, -1.0], [5.0, 0.0], [6.0, 1.0],
                [7.0, 2.0], [8.0, 3.0], [9.0, 4.0]]

    y_treino = [1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0, -1.0, -1.0,
                -1.0, -1.0, -1.0]

    x_classificar = [[12.0, 5.0], [7.0, 0.0], [-2.0, -9.0], [2.0, -13.0], [-5.0, 4.0],
                     [-11.0, 6.0], [5.0, -2.0], [-7, 5.0], [6.0, 13.0], [0.0, 8.0]]

    nn = Perceptron(0.01, 50, 2)
    nn.exibir_dados_da_rede()
    nn.treinar(x_treino, y_treino)
    nn.exibir_dados_da_rede()
    y_classificado = nn.classificar(x_classificar, classe_a="Azul", classe_b="Vermelha")

    if bool(y_classificado):
        for amostra in range(len(y_classificado)):
            print("Saída prevista para a amostra {0}: {1}".format(amostra + 1, y_classificado[amostra]))

    plotar_dados_de_treinamento(x_treino, y_treino, nn.w, x_classificar)


def plotar_dados_de_treinamento(ent_treino, sai_treino, pesos_w, ent_class):

    eixo_x_treino = []
    eixo_y_treino = []
    cores_par_x_y_treino = []

    # Separa os pontos x e y do conjunto de dados de treinamento em duas listas diferentes
    for par_x_y in ent_treino:
        eixo_x_treino.append(par_x_y[1])
        eixo_y_treino.append(par_x_y[2])

    # Cria uma lista que define a cor de cada ponto do conjunto de dados (baseado na saída y)
    for par_x_y in sai_treino:
        if par_x_y == -1:
            cores_par_x_y_treino.append("b")
        elif par_x_y == 1:
            cores_par_x_y_treino.append("r")

    # Plota o conjunto de dados de treinamento em vermelho(1) e azul(-1):
    tamanho = 50
    plt.scatter(eixo_x_treino, eixo_y_treino, s=tamanho, c=cores_par_x_y_treino)

    # Plota a função que melhor se adequa ao conjunto de dados de treinamento em preto:
    x = range(-15, 16, 1)
    plt.plot(x, x, c="k")

    # Plota a função encontrada pela rede neural que representa a relação dos dados em amarelo:
    x1 = range(-15, 16, 1)
    x2 = []
    for value in x1:
        x2.append((pesos_w[0] - (value * pesos_w[1])) / pesos_w[2])
    plt.plot(x1, x2, c="y", linestyle="-.")

    eixo_x_classificacao = []
    eixo_y_classificacao = []
    cores_par_x_y_class = []

    # Separa os pontos x e y do conjunto de dados de classificação em duas listas diferentes
    for par_x_y in ent_class:
        eixo_x_classificacao.append(par_x_y[1])
        eixo_y_classificacao.append(par_x_y[2])

    for i in range(len(ent_class)):
        cores_par_x_y_class.append("grey")

    # Plota o conjunto de dados para classificação em cinza:
    tamanho = 100
    plt.scatter(eixo_x_classificacao, eixo_y_classificacao, s=tamanho, c=cores_par_x_y_class, marker="*")

    # Exibe o gráfico
    plt.grid(True)
    plt.show()
