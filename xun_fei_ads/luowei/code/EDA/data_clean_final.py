import re
import pandas as pd
import urllib.parse
import numpy as np
df_model=pd.read_csv('needs_to_clean.csv',dtype=str)

pattern1=r'%2522|%(25){0,}2B|%20|%(25){0,}28|%(25){0,}29|%(25){0,}2C'
#用空格替代连字符"-或者+或者_"
pattern2=r'-{1,}|(?<=[a-zA-Z\d])\+{1,}\s{0,}(?=[a-zA-Z\d])|_{1,}'
#两个以上的空格都替换成一个
pattern3='\s{2,}'
#去掉括号
pattern4='\(|\)'

def reg(input_string):
    if isinstance(input_string, str):
        new_string=re.sub(pattern1,' ',input_string)
        new_string=urllib.parse.unquote(new_string)
        new_string = re.sub(pattern2,' ',new_string)
        new_string = re.sub(pattern3,' ', new_string)
        new_string = re.sub(pattern4,'', new_string)
        # 全部转为大写
        new_string = new_string.upper()
        if '%' in  new_string:
            print(input_string,new_string)
            new_string=np.nan
        return new_string
    else:
        # print(input_string)
        return input_string

df_model['make_new']=df_model.apply(lambda row:reg(row['make']),axis=1)
df_model['model_new']=df_model.apply(lambda row:reg(row['model']),axis=1)
df_model.to_csv('cleaned_data_final.csv',index=None)
# df_model.to_excel('cleaned_data_final.xls',index=None)