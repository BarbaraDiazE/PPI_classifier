import pandas as pd

def dos(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    return df1, df2

def tres(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    df3 =  pd.read_csv(archivos[2])
    return df1, df2, df3

def cuatro(*archivos):
    df1 = pd.read_csv(archivos[0], sep = "\t")
    df2 =  pd.read_csv(archivos[1], sep = "\t")
    df3 =  pd.read_csv(archivos[2], sep = "\t")
    df4 =  pd.read_csv(archivos[3], sep = "\t")
    return df1, df2, df3, df4

def cinco(*archivos):
    df1 = pd.read_csv(archivos[0], sep = ",")
    df2 =  pd.read_csv(archivos[1], sep = ",")
    df3 =  pd.read_csv(archivos[2], sep = ",")
    df4 =  pd.read_csv(archivos[3], sep = ",")
    df5 =  pd.read_csv(archivos[4], sep = ",")
    return df1, df2, df3, df4, df5

def seis(*archivos):
    df1 = pd.read_csv(archivos[0], sep = "\t")
    df2 =  pd.read_csv(archivos[1], sep = "\t")
    df3 =  pd.read_csv(archivos[2], sep = "\t")
    df4 =  pd.read_csv(archivos[3], sep = "\t")
    df5 =  pd.read_csv(archivos[4], sep = "\t")
    df6 =  pd.read_csv(archivos[5], sep = "\t")
    return df1, df2, df3, df4, df5, df6

def siete(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    df3 =  pd.read_csv(archivos[2])
    df4 =  pd.read_csv(archivos[3])
    df5 =  pd.read_csv(archivos[4])
    df6 =  pd.read_csv(archivos[5])
    df7 =  pd.read_csv(archivos[6])
    return df1, df2, df3, df4, df5, df6, df7

def ocho(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    df3 =  pd.read_csv(archivos[2])
    df4 =  pd.read_csv(archivos[3])
    df5 =  pd.read_csv(archivos[4])
    df6 =  pd.read_csv(archivos[5])
    df7 =  pd.read_csv(archivos[6])
    df8 =  pd.read_csv(archivos[7])
    return df1, df2, df3, df4, df5, df6, df7, df8

def nueve(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    df3 =  pd.read_csv(archivos[2])
    df4 =  pd.read_csv(archivos[3])
    df5 =  pd.read_csv(archivos[4])
    df6 =  pd.read_csv(archivos[5])
    df7 =  pd.read_csv(archivos[6])
    df8 =  pd.read_csv(archivos[7])
    df9 =  pd.read_csv(archivos[8])
    return df1, df2, df3, df4, df5, df6, df7, df8, df9

def diez(*archivos):
    df1 = pd.read_csv(archivos[0])
    df2 =  pd.read_csv(archivos[1])
    df3 =  pd.read_csv(archivos[2])
    df4 =  pd.read_csv(archivos[3])
    df5 =  pd.read_csv(archivos[4])
    df6 =  pd.read_csv(archivos[5])
    df7 =  pd.read_csv(archivos[6])
    df8 =  pd.read_csv(archivos[7])
    df9 =  pd.read_csv(archivos[8])
    df10 =  pd.read_csv(archivos[9])
    print( df1.shape[0], df2.shape[0], df3.shape[0], df4.shape[0], df5.shape[0], df6.shape[0], df7.shape[0], df8.shape[0], df9.shape[0], df10.shape[0])
    return df1, df2, df3, df4, df5, df6, df7, df8, df9, df10

