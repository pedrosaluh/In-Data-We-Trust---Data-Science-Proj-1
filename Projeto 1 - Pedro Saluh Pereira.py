#!/usr/bin/env python
# coding: utf-8

# **TÍTULO: Projeto 1 - HOTEL BOOKING DEMAND**
# 
# **AUTOR: Pedro Saluh Pereira**
# 
# **DESCRIÇÃO DOS DADOS:** Esse db contém informações sobre reservas de diferentes tipos de hotel (cidade vs resort), como por exemplo, data da reserva, estadia, número de hóspedes e etc.
# 

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[30]:


#Importando a base de dados hotel_bookings
hotel_bookings = pd.read_csv ('G:\Meu Drive\In Data we trust\DATA SCIENCE\Projeto 1\hotel_bookings.csv')


# **CONHECENDO A BASE DE DADOS**

# In[31]:


#Entendendo quais são os dados que tem na base de dados
hotel_bookings.sample(5)


# **LIMPEZA E TRATAMENTO DE DADOS:**

# **TRATAMENTO DO TIPO DE VARIÁVEL**

# In[32]:


#Entendendo algumas informações gerais do banco de dados
hotel_bookings.info()


# In[33]:


#Verificando os tipos de variáveis no banco de dados
for var in hotel_bookings:
    # imprime variavel e seu tipo
    print(var,":", hotel_bookings[var].dtype.name, end="")
    # se nao numérico
    if not np.issubdtype(hotel_bookings[var].dtype, np.number):
        print("\n\t",hotel_bookings[var].nunique(), "distintos: ", end="")
        print(hotel_bookings[var].unique())
    else:
        print(", intervalo: ",end="")
        print(hotel_bookings[var].min(), ",", hotel_bookings[var].max())


# A variável 'children', que deveria ser inteira, está como float. Precisamos converter:

# In[34]:


#Convertendo a variável para inteiro.
hotel_bookings['children'] = (round(hotel_bookings['children'])).values.astype(np.int64)

hotel_bookings.info()


# **DADOS FALTANTES**
# 
# Entendendo quais são os dados faltantes para que completemos ou eliminemos os dados.

# In[35]:


#Verificando se alguma coluna tem informações faltantes:
hotel_bookings.isna().sum()


# Temos 4 colunas com algum dado faltante:
# 
# Children = 4
# 
# Country = 488
# 
# Agent = 16340
# 
# Company = 112593
# 
# A coluna "company" tem a maior parte dos dados nulo, então é melhor excluí-la:

# In[36]:


#Excluindo a coluna "company" da BD
hotel_bookings_limpa = hotel_bookings.drop(['company'], axis=1)

#Confirmando se a coluna foi excluida
hotel_bookings_limpa.info()


# Para as demais, precisamos escolher se vamos preencher os dados nulos, ou se vamos excluir as linhas.
# 
# Para a coluna "agent", se excluirmos as linhas com dados nulos, perderemos muitos dados. Por isso, vamos preencher os valores nulos, usando como base as informações nas linhas que não são nulas. Apesar de os valores serem numéricos (float), eles são IDs das agências e, por isso, não faz sentido calcular uma média das ocorrências ou algo do tipo. Por isso, vamos ver se há alguma agência que se repete mais e, se houver, vamos usá-la.

# In[37]:


display(hotel_bookings['agent'].value_counts())


# A agência 9.0 é, de longe, a que mais aparece. Por isso, vamos substituir os dados nulos por ela:

# In[38]:


hotel_bookings_limpa["agent"].fillna(9, inplace=True)


# In[39]:


#Confirmando se os valores nulos foram substituídos
hotel_bookings_limpa.isna().sum()


# Os dados faltantes nas colunas "country" e "children" são poucos. Por isso, vamos excluir as linhas em que estes valores não aparecem, pois não vamos perder muita massa de dados:

# In[40]:


hotel_bookings_limpa = hotel_bookings_limpa.dropna()


# In[41]:


#Confirmando se os valores nulos foram excluídos
hotel_bookings_limpa.isna().sum()


# **REMOVENDO DADOS DUPLICADOS**

# In[42]:


#Imprimindo dados duplicados
hotel_bookings_limpa[hotel_bookings_limpa.duplicated()]


# In[43]:


#Removendo linhas duplicadas, mas mantendo a primeira aparição
hotel_bookings_limpa2 = hotel_bookings_limpa.drop_duplicates(keep='first')

#Atualizando informações da lista para garantir que duplicadas foram excluídas
hotel_bookings_limpa2.info()


# **REMOVENDO OUTLIERS**
# 
# A título de exemplo, vamos identificar e remover outliers apenas dos atributos "days_in_waiting_list" e "lead_time".

# In[44]:


#BOXPLOT de dias de espera

plt.figure(figsize=(15,8))

sns.boxplot(x="customer_type", y="days_in_waiting_list", hue="hotel", data=hotel_bookings_limpa2, palette="Set3")

plt.show()


# In[45]:


#Função para remover outliers pela metodologia IQR

def remove_outliers_IQR(df, attributes, factor=2):
    """Funcao para remover outliers com base no IQR
    Parametros:
        - df : dataframe
        - attributes: atributos a considerar na remoção
        - factor: fator do IQR a considerar
    Retorno:
        dataframe com os outliers removidos
    """
    dfn = df.copy()
    
    for var in attributes:
        # verifica se variável é numerica
        if np.issubdtype(df[var].dtype, np.number):
            Q1 = dfn[var].quantile(0.25)
            Q2 = dfn[var].quantile(0.50)
            Q3 = dfn[var].quantile(0.75)
            IQR = Q3 - Q1
            
            # apenas inliers segundo IQR
            dfn = dfn.loc[(df[var] >= Q1-(IQR*factor)) & (df[var] <= Q3+(IQR*factor)),:]

    return dfn


# In[46]:


#Removendo outliers do atributo days_in_waiting_list

hotel_bookings_limpa3 = remove_outliers_IQR(hotel_bookings_limpa2, ['days_in_waiting_list'])


# In[47]:


#Verificando como a nova base ficou
hotel_bookings_limpa3.info()


# In[48]:


#BOXPLOT de lead time

plt.figure(figsize=(15,8))

sns.boxplot(x="customer_type", y="lead_time", hue="hotel", data=hotel_bookings_limpa2, palette="Set3")

plt.show()


# In[49]:


#Removendo outliers do atributo lead_time
hotel_bookings_limpa3 = remove_outliers_IQR(hotel_bookings_limpa3, ['lead_time'])


# In[50]:


#Verificando como a nova base ficou
hotel_bookings_limpa3.info()


# **ANÁLISE EXPLORATÓRIA DE DADOS**

# In[73]:


#Plotando gráfico para entender a quantidade de clientes por tipo, para cada perfil de hotel

plt.figure(figsize = (15,8))

sns.countplot(data = hotel_bookings_limpa3, x='customer_type', hue = 'hotel')

plt.title('Customer type x Hotel', fontsize = 14)

plt.xlabel('Reservas', fontsize = 10)
plt.ylabel('Tipo de cliente', fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

plt.tight_layout()

plt.show()


# **Conclusão:**
# 
# 1 - O tipo de cliente mais comum é o "transient".
# 
# 2 - Clientes "transient" e "transient-party" são mais comuns em hotéis de cidade, enquanto os "contract" e "group" são mais comuns em resorts.

# In[118]:


#Plotando gráfico para entender se há alguma sazonalidade no número de reservas

new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
             'October', 'November', 'December']

plt.figure(figsize = (15,8))

plt.title('Quantidade de reservas por mês', fontsize = 14)

sns.countplot(data = hotel_bookings_limpa3, x='arrival_date_month')
plt.show()


# **Conclusão:** Há sim uma sazonalidade, com picos entre julho e agosto. 

# In[68]:


#Plotando gráfico para entender se a sazonalidade no número de reservas muda de acordo com o perfil do hotel

# Cria o FacetGrid
grid = sns.FacetGrid(hotel_bookings_limpa3, col='hotel')


# Cria o gráfico relacionado
grid.map(sns.histplot, 'arrival_date_month')

grid = grid.fig.set_size_inches(20,10)

# Mostra o Gráfico
plt.show()


# **Conclusão**: Enquanto em resorts a sazonalidade é bem desenhada, com picos claros em Julho e Agosto, nos hotéis de cidade há também um pico nestes meses, mas há também um volume representativo de reservas entre março e junho.

# In[117]:


#Plotando gráfico para entender se a sazonalidade se manteve ao longo dos anos

# Cria o FacetGrid
grid = sns.FacetGrid(hotel_bookings_limpa3, col='arrival_date_year')


# Cria o gráfico relacionado
grid.map(sns.histplot, 'arrival_date_month')

grid = grid.fig.set_size_inches(20,10)

# Mostra o Gráfico
plt.show()


# **Conclusão:** O ano de 2017 parece estar se comportando diferente do ano de 2016. Enquanto em 2016 tivemos saltos grandes de janeiro para fevereiro e de fevereiro para março, com estabilidade entre março e maio, em 2017 as reservas vão subindo gradualmente desde janeiro, até chegar num patamar de estabilidade em maio.

# In[129]:


#Plotando gráfico para entender os tipos de refeição mais consumidos

# Cria o displot
sns.displot(hotel_bookings_limpa3["meal"])


#mostra o Gráfico
plt.show()


# **Conclusão**: A refeição FB tem baixo consumo. Poderia ser eliminada ou substituída.

# In[141]:


#Plotando gráfico para entender a proporção entre reservas canceladas e reservas não canceladas

plt.figure(figsize = (15,8))

sns.countplot(data = hotel_bookings_limpa3, x='is_canceled')

plt.title('Reservas canceladas vs Não canceladas', fontsize = 14)

plt.xlabel('Cancelada vs Não cancelada', fontsize = 10)
plt.ylabel('Qtde', fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

plt.tight_layout()

plt.show()


# **Conclusão**: Temos um número representativo de reservas canceladas!

# In[142]:


#Plotando gráfico para entender como se dá a proporção entre reservas canceladas e reservas não canceladas, por tipo de cliente

plt.figure(figsize = (15,8))

sns.countplot(data = hotel_bookings_limpa3, x='customer_type', hue = 'is_canceled')

plt.title('Cancelamento de reservas x Customer type', fontsize = 14)

plt.xlabel('Tipo de cliente', fontsize = 10)
plt.ylabel('Reservas', fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

plt.tight_layout()

plt.show()


# **Conclusão:** A taxa proporcional de cancelamento dos clientes "transient" são muito maiores que as dos demais tipos de cliente.

# In[143]:


#Plotando gráfico para entender como se dá a proporção entre reservas canceladas e reservas não canceladas, para cada perfil de hotel

plt.figure(figsize = (15,8))

sns.countplot(data = hotel_bookings_limpa3, x='hotel', hue = 'is_canceled')

plt.title('Cancelamento de reservas x Tipo de hotel', fontsize = 14)

plt.xlabel('Tipo de hotel', fontsize = 10)
plt.ylabel('Reservas', fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

plt.tight_layout()

plt.show()


# **Conclusão:** A taxa proporcional de cancelamento é maior nos hotéis de cidade do que nos resorts.

# In[145]:


#Plotando gráfico para entender como se dá a proporção entre reservas canceladas e reservas não canceladas, por mês
plt.figure(figsize = (15,8))

sns.countplot(data = hotel_bookings_limpa3, x='arrival_date_month', hue = 'is_canceled')

plt.title('Cancelamento de reservas x Mês', fontsize = 14)

plt.xlabel('Mês', fontsize = 10)
plt.ylabel('Reservas', fontsize = 10)

plt.xticks(fontsize = 8)
plt.yticks(fontsize = 8)

plt.tight_layout()

plt.show()


# In[150]:


#Plotando gráfico para entender a duração das reservas por tipo de cliente e por tipo de hotel

plt.figure(figsize=(15,8))

sns.boxplot(x="customer_type", y="stays_in_week_nights", hue="hotel", data=hotel_bookings_limpa2, palette="Set3")

plt.show()


# **Conclusão:**
# 
# 1 - Hotéis de cidade tem estadias menos duradouras que resorts, para todos os tipos de cliente
# 
# 2 - Para clientes do tipo "contract", a diferença na duração das estadias entre resort e hotéis de cidade é ainda maior
# 
# 3 - As durações de estadias dos clientes "contract" de resorts, em geral, são muito maiores que as demais, mesmo que em alguns casos isolados tenhamos estadias mais duradouras, como por exemplo, uma estadia de 40 dias nos clientes "transient" do resort.
# 
# 4 - Desconsiderando o cliente "group", por ter poucas amostras, os casos que tem uma dispersão menor entre as durações das estadias é nos clientes "contract" de hotéis de cidade, representando uma previsibilidade maior da duração das estadias. O grupo que tem menos previsibilidade sobre a duração das estadias é o de "transient" de resorts.

# In[ ]:




