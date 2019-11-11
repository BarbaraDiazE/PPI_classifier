import pandas as pd

def dos(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    return df1, df2

def tres(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    return df1, df2, df3

def cuatro(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    return df1, df2, df3, df4

def cinco(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    df5 =  pd.read_csv(str(root) + archivos[4], sep = sep)
    return df1, df2, df3, df4, df5

def seis(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    df5 =  pd.read_csv(str(root) + archivos[4], sep = sep)
    df6 =  pd.read_csv(str(root) + archivos[5], sep = sep)
    return df1, df2, df3, df4, df5, df6

def siete(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    df5 =  pd.read_csv(str(root) + archivos[4], sep = sep)
    df6 =  pd.read_csv(str(root) + archivos[5], sep = sep)
    df7 =  pd.read_csv(str(root) + archivos[6], sep = sep)
    return df1, df2, df3, df4, df5, df6, df7

def ocho(root, sep, *archivos):
    df1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    df5 =  pd.read_csv(str(root) + archivos[4], sep = sep)
    df6 =  pd.read_csv(str(root) + archivos[5], sep = sep)
    df7 =  pd.read_csv(str(root) + archivos[6], sep = sep)
    df8 =  pd.read_csv(str(root) + archivos[7], sep = sep)
    return df1, df2, df3, df4, df5, df6, df7, df8

def nueve(root, sep, *archivos):
    f1 = pd.read_csv(str(root) + archivos[0], sep = sep)
    df2 =  pd.read_csv(str(root) + archivos[1], sep = sep)
    df3 =  pd.read_csv(str(root) + archivos[2], sep = sep)
    df4 =  pd.read_csv(str(root) + archivos[3], sep = sep)
    df5 =  pd.read_csv(str(root) + archivos[4], sep = sep)
    df6 =  pd.read_csv(str(root) + archivos[5], sep = sep)
    df7 =  pd.read_csv(str(root) + archivos[6], sep = sep)
    df8 =  pd.read_csv(str(root) + archivos[7], sep = sep)
    df9 =  pd.read_csv(str(root) + archivos[8], sep = sep)
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

