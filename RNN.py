'''
    Para o emprego de uma rede neural artificial feedforward como preditor de um passo à frente, 
    é necessário definir quais valores passados da série serão utilizados na definição da entrada da 
    rede neural. Feito isso, o problema de síntese do preditor se transforma em um problema de 
    treinamento supervisionado, onde o que se deseja é obter um mapeamento multidimensional 
    não-linear de entrada-saída, como indicado na sequencia de passos abaixo.
'''

import numpy as np
import math
import matplotlib.pyplot as plt


'''
geradados: Esta função gera os dados para o treinamento, validação e teste da rede. 
 Ela recebe uma série temporal, a quantidade de lag (retardos a serem considerados), o passo a frente para a 
 previsão, o número de dados de treino, validação e teste, e retorna os conjuntos de dados para cada etapa.
'''
def geradados(open, lag, passo, Ntr, Nval, Ntest):
    x = []
    y = []
    N = len(open)
    
    for i in range(N):
        x.append(open[i:lag+i])
        y.append(open[lag+i+passo])
    
    x = np.array(x)
    y = np.array(y)
    Xtr = x[:Ntr, :]
    Xval = x[Ntr:Ntr+Nval, :]
    Xtest = x[Ntr+Nval:, :]
    Ytr = y[:Ntr]
    Yval = y[Ntr:Ntr+Nval]
    Ytest = y[Ntr+Nval:]
    
    return Xtr, Xval, Xtest, Ytr, Yval, Ytest


'''
treina_mlp: Esta função realiza o treinamento da MLP. Ela recebe os dados de entrada
 e saída desejada para o treinamento e validação, o número de neurônios na camada escondida 
 e o número máximo de épocas. As matrizes de peso são inicializadas aleatoriamente 
 e o loop de treinamento é executado até que o gradiente seja menor do que um valor pequeno (1e-5) 
 ou o número máximo de épocas seja atingido. A cada época, o algoritmo calcula o alfa ótimo (taxa de aprendizado) 
 usando a função 'calc_alfa', atualiza os pesos e calcula o erro quadrático médio (EQM). 
 Se o EQM na validação diminui, os pesos são atualizados. A função retorna os pesos que minimizam o EQM na validação.
'''
def treina_mlp(X, Yd, Xval, Ydval, h, nepocasmax):
    N, ne = X.shape
    Nval = Xval.shape[0]
    X = np.hstack((X, np.ones((N, 1))))
    Xval = np.hstack((Xval, np.ones((Nval, 1))))
    ns = Yd.shape[1]
    A = np.random.rand(h, ne+1)
    B = np.random.rand(ns, h+1)
    Yr = calc_saida(X, A, B, h, N)
    dJdA, dJdB = calc_grad(X, Yd, A, B, N)
    grad = np.hstack((dJdA.ravel(), dJdB.ravel()))
    erro = Yr - Yd
    EQM = 1/N*np.sum(erro**2)
    Yrval = calc_saida(Xval, A, B, h, Nval)
    erroval = Yrval - Ydval
    EQMval = 1/N*np.sum(erroval**2)
    vetEQM = []
    vetEQM.append(EQM)
    nep = 0
    
    while np.linalg.norm(grad) > 1e-5 and nep < nepocasmax:
        nep += 1
        alfa = calc_alfa(X, Yd, A, B, dJdA, dJdB, N)
        A = A - alfa*dJdA
        B = B - alfa*dJdB
        Yr = calc_saida(X, A, B, h, N)
        dJdA, dJdB = calc_grad(X, Yd, A, B, N)
        grad = np.hstack((dJdA.ravel(), dJdB.ravel()))
        erro = Yr - Yd
        EQM = 1/N*np.sum(erro**2)
        vetEQM.append(EQM)
        Yrval = calc_saida(Xval, A, B, h, Nval)
        erroval = Yrval - Ydval
        EQMval_new = 1/N*np.sum(erroval**2)
        if EQMval_new < EQMval:
            EQMval = EQMval_new
            Aval = A
            Bval = B
    
    plt.plot(vetEQM)
    print(vetEQM[-1])
    return Aval, Bval

'''
calc_saida: Esta função calcula a saída da rede para uma dada entrada X e pesos A e B. 
 Ela usa a função logística como função de ativação.
'''
def calc_saida(X, A, B, h, N):
    Zin = np.dot(X, A.T)
    Z = 1 / (1 + np.exp(-Zin))
    Zb = np.hstack((Z, np.ones((N, 1))))
    Yin = np.dot(Zb, B.T)
    Y = 1 / (1 + np.exp(-Yin))
    return Y

'''
calc_grad: Esta função calcula o gradiente da função de erro. Ela recebe os dados 
 de entrada e saída desejada, os pesos e o número de exemplos, e retorna os gradientes com respeito a A e B.
'''
def calc_grad(X, Yd, A, B, N):
    Zin = np.dot(X, A.T)
    Z = 1 / (1 + np.exp(-Zin))
    Zb = np.hstack((Z, np.ones((N, 1))))
    Yin = np.dot(Zb, B.T)
    Y = 1 / (1 + np.exp(-Yin))
    erro = Y - Yd
    fl = (1 - Z) * Z
    gl = (1 - Y) * Y
    dJdB = 1 / N * np.dot((erro * gl).T, Zb)
    dJdZ = np.dot(erro * gl, B[:, :B.shape[1] - 1])
    dJdA = 1 / N * np.dot((dJdZ * fl).T, X)
    return dJdA, dJdB

'''
calc_alfa: Esta função calcula a taxa de aprendizado ótima para a atualização dos pesos.
 Ela recebe os dados de entrada e saída desejada, os pesos, os gradientes e o número de exemplos,
 e retorna a taxa de aprendizado.
'''
def calc_alfa(X, Yd, A, B, dJdA, dJdB, N):
    d = np.vstack((-dJdA.flatten(), -dJdB.flatten()))
    alfa_l = 0
    alfa_u = np.random.rand()
    Aaux = A - alfa_u * dJdA
    Baux = B - alfa_u * dJdB
    dJdAaux, dJdBaux = calc_grad(X, Yd, Aaux, Baux, N)
    grad = np.vstack((dJdAaux.flatten(), dJdBaux.flatten()))
    hl = np.dot(grad.T, d)

    while hl < 0:
        alfa_l = alfa_u
        alfa_u = 2 * alfa_u
        Aaux = A - alfa_u * dJdA
        Baux = B - alfa_u * dJdB
        dJdAaux, dJdBaux = calc_grad(X, Yd, Aaux, Baux, N)
        grad = np.vstack((dJdAaux.flatten(), dJdBaux.flatten()))
        hl = np.dot(grad.T, d)

    alfa_m = (alfa_l + alfa_u) / 2
    Aaux = A - alfa_m * dJdA
    Baux = B - alfa_m * dJdB
    dJdAaux, dJdBaux = calc_grad(X, Yd, Aaux, Baux, N)
    grad = np.vstack((dJdAaux.flatten(), dJdBaux.flatten()))
    hl = np.dot(grad.T, d)

    k = 0
    kmax = math.ceil(np.log((alfa_u - alfa_l) / 1e-4))
    while k < kmax and abs(hl) > 1e-4:
        if hl > 0:
            alfa_u = alfa_m
        elif hl < 0:
            alfa_l = alfa_m
        else:
            break

        alfa_m = (alfa_l + alfa_u) / 2
        Aaux = A - alfa_m * dJdA
        Baux = B - alfa_m * dJdB
        dJdAaux, dJdBaux = calc_grad(X, Yd, Aaux, Baux, N)
        grad = np.vstack((dJdAaux.flatten(), dJdBaux.flatten()))
        hl = np.dot(grad.T, d)
        k += 1

    return alfa_m



