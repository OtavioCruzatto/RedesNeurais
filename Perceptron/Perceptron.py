import random


def inserir_valor_unitario_em_x0(x):
    """ Adiciona o valor -1 na posição 0 de cada amostra de entrada."""
    for item in x:
        item.insert(0, -1)


def verificar_qtd_de_dados(x, y):
    """ Verifica se a quantidade de amostras de entrada é igual a quantidade de amostras de saída."""
    if len(x) != len(y):
        print("Não foi possível treinar a rede.")
        print("Quantidade de dados de entrada diferente da quantidade de dados de saida.")
        print("Entrada: {0} amostras".format(len(x)))
        print("Saida: {0} amostras".format(len(y)))
        return False
    return True


def verificar_dados_de_saida(y):
    """ Verifica se os dados de saída estão formatados corretamente."""
    # Testa se o conjunto de dados de saida está vazio
    if not bool(y):
        print("Não foi possível treinar a rede.")
        print("Conjunto de dados de saida está vazio.")
        return False
    # Testa se cada amostra de saída possui apenas um valor
    for valor in y:
        if (type(valor) is not int) and (type(valor) is not float):
            print("Não foi possível treinar a rede.")
            print("O conjunto de dados de saída possui valor(es) que não são int's ou float's.")
            return False
    # Testa se as amostras de saída estão no formato correto (apenas valores 1 e -1)
    for valor in y:
        if (valor != 1) and (valor != -1):
            print("Não foi possível treinar a rede.")
            print("O conjunto de dados de saída possui valor(es) diferentes de 1 e -1.")
            return False
    return True


def verificar_dados_de_entrada(x, qtd_entr):
    """ Verifica se os dados de entrada estão formatados corretamente."""
    # Testa se o conjunto de dados de entrada está vazio
    for item in x:
        if not bool(item):
            print("Não foi possível treinar a rede.")
            print("Conjunto de dados de entrada está vazio.")
            return False
    # Testa se cada amostra de entrada possui a quantidade "qtd_entr" especificada na criação da rede
    for item in x:
        if len(item) != qtd_entr:
            print("Não foi possível treinar a rede.")
            print(
                "O conjunto de dados de entrada possui itens que contém quantidades de dados "
                "diferente da quantidade de entradas da rede.")
            return False
    return True


def validar_dados_de_treinamento(x, y, qtd_entr):
    """ Verifica se os dados de treinamento estão formatados corretamente."""
    validacao_1 = verificar_qtd_de_dados(x, y)
    validacao_2 = verificar_dados_de_entrada(x, qtd_entr)
    validacao_3 = verificar_dados_de_saida(y)
    if validacao_1 and validacao_2 and validacao_3:
        return True
    else:
        return False


def degrau_bipolar(u):
    """ Função de ativação - Degrau bipolar"""
    if u >= 0:
        return 1
    else:
        return -1


def degrau_unitario(u):
    """ Função de ativação - Degrau unitário"""
    if u >= 0:
        return 1
    else:
        return 0


class Perceptron:
    """ Classe Perceptron."""

    def __init__(self, n, ep_max, qtd_entr):
        """ Construtor da classe Perceptron."""

        self.n = n
        """ Taxa de aprendizado da rede (n = eta)"""

        self.ep_max = ep_max
        """ Quantidade máxima de épocas permitidas"""

        self.ep_atual = 0
        """ Quantidade de épocas de treinamento atual"""

        self.qtd_entr = qtd_entr
        """ Quantidade de entradas da rede"""

        self.treinada = False
        """ Indicação se a rede já foi treinada"""

        self.w = []
        """ Vetor de pesos sinápticos (inicializado randomicamente com valores entre 0 e 1)"""

        for i in range(self.qtd_entr + 1):
            self.w.append(random.random())

    def exibir_dados_da_rede(self):
        """ Exibe informações referentes a rede neural Perceptron."""

        print("===========================================")
        print("\nStatus da rede: ", end="")
        if self.treinada:
            print("Rede treinada!")
        else:
            print("Rede não treinada!")

        print("\nÉpocas de treinamento (qtd de vezes que todo o conjunto de treinamento foi apresentado para a rede):")
        print("Quantidade máx: {0}".format(self.ep_max))
        print("Quantidade atual: {0}".format(self.ep_atual))

        print("\nPesos sinápticos:")
        for i in range(len(self.w)):
            print("w[{0}] = {1}".format(i, self.w[i]))

        print("\nTaxa de aprendizado: {0}".format(self.n))
        print("Quantidade de entradas da rede: {0}".format(self.qtd_entr))
        print("\n===========================================\n")

    def treinar(self, x, y):
        """ Realiza o treinamento da rede com os dados de treinamento x e y (entrada e saída)"""

        print("Treinando...")

        # Realizar a validação dos dados de treinamento:
        if validar_dados_de_treinamento(x, y, self.qtd_entr):

            # Add o valor -1 na posição 0 de cada amostra de entrada
            inserir_valor_unitario_em_x0(x)

            while True:

                # Atualização de variáveis
                erro = False
                self.ep_atual = self.ep_atual + 1

                # Para cada amostra de treinamento:
                for amostra_treinamento in range(len(x)):

                    # Calcula potencial de ativação u:
                    u = 0
                    for entrada in range(self.qtd_entr + 1):
                        u = u + (x[amostra_treinamento][entrada] * self.w[entrada])

                    # Aplica a função de ativação sobre o potencial de ativação (prever saída):
                    y_previsto = degrau_bipolar(u)

                    # Verfica se a saída prevista pela rede está incorreta (comparando com a amostra de treinamento):
                    if y_previsto != y[amostra_treinamento]:
                        erro = True

                        # Caso exita erro, ajusta os pesos sinápticos:
                        for entrada in range(self.qtd_entr + 1):
                            self.w[entrada] = self.w[entrada] + (
                                    self.n * (y[amostra_treinamento] - y_previsto) * x[amostra_treinamento][entrada])

                # Verifica se a última época de treinamento não apresentou erros de predição.
                if not erro:
                    print("Treinamento realizado!\n")
                    self.treinada = True
                    break

                # Verifica se o treinamento alcançou a quantidade máx de épocas permitida.
                if self.ep_atual >= self.ep_max:
                    print("Treinamento abortado!")
                    print("Quantidade máxima de épocas de treinamento atingida!\n")
                    break

        else:
            print("Treinamento abortado!")
            print("Erro na validação dos dados!\n")

    def classificar(self, x, classe_a="A", classe_b="B"):
        """ Utiliza os pesos ajustados durante o treinamento para classificar novas entradas."""

        print("Classificando...")

        # Testa se a rede já foi treinada
        if not self.treinada:
            print("Classificação abortada!")
            print("A rede ainda não foi treinada, realize o treinamento antes de fazer uma classificação!\n")
            return None
        else:

            # Cria lista de saída:
            saida_prevista = []

            # Add o valor -1 na posição 0 de cada amostra de entrada
            inserir_valor_unitario_em_x0(x)

            # Para cada amostra de classificação:
            for amostra_classificacao in range(len(x)):

                # Calcula potencia de ativação:
                u = 0
                for entrada in range(self.qtd_entr + 1):
                    u = u + x[amostra_classificacao][entrada] * self.w[entrada]

                # Aplica a função de ativação sobre o potencial de ativação (prever saída):
                y_previsto = degrau_bipolar(u)

                if y_previsto == -1:
                    saida_prevista.append(classe_a)
                elif y_previsto == 1:
                    saida_prevista.append(classe_b)

            print("Classificação realizada!\n")

            # Retorna a lista de saídas previstas:
            return saida_prevista
