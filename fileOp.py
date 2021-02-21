# I/O tools
import pandas as pd
import os.path

def reIndex(df):
    # recombine 'row 1 (row 2)' as new header including units
    df.columns = df.columns.map(lambda h: '{} ({})'.format(h[0], h[1]))
    return df


def readTxtDelimiter(fileName, delimiter='[step]', title=[0], header=[1,2]):
    # read a huge text file with delimiter to separate sheets, and output to a new xlsx file; title and header correspond to sheet name and header row numbers 
    from io import StringIO 
    if os.path.exists(fileName+'.xlsx'):
        print(fileName+'.xlsx file exists. Do nothing.')
    else:
        print(fileName+'.xlsx file does not exist. Creating now...')
        writer = pd.ExcelWriter(fileName+'.xlsx')
        with open(fileName+'.txt') as fp:
            contents = fp.read()
            for entry in contents.split(delimiter):
                stringData = StringIO(entry)
                df = pd.read_csv(stringData, sep="\t", skiprows=title, header=header)
                df.to_excel(writer, entry.split('\n')[title[0]+1])
        writer.save()
        
def readXLS(fileName, reindex=False, appTag=None, *args, **kwargs):
    # read from a xls sheet with the same arguments as read_excel
    if os.path.exists(fileName+'.xlsx') or os.path.exists(fileName+'.xls'):
        if os.path.exists(fileName+'.xlsx'):
            p = pd.read_excel(fileName+'.xlsx', *args, **kwargs)
        else:
            p = pd.read_excel(fileName+'.xls', *args, **kwargs)
        
        if reindex:
            p = reIndex(p)
            
        if appTag is not None: 
            for tag, value in appTag.items():
                p[tag] = value
                
        return p
        
    else:
        print('No '+fileName+'.xls file. Please run readTxtDelimiter(fileName) first or generate from the original software.')
