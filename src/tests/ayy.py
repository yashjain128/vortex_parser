import openpyxl  

def func(*args):
    print(*args)


wb = openpyxl.load_workbook('lib/SAILFormat.xlsx', data_only=True)  
sh = wb.active

getval = lambda c: str(sh[c].value)

for row in sh[ 'C'+getval('C3') : 'H'+getval('D3')]: 
    func(*[i.value for i in row])

for row in sh[ 'C'+getval('C4') : 'O'+getval('D4')]: 
    func(*[i.value for i in row])

for row in sh[ 'C'+getval('C5') : 'U'+getval('D5')]: 
    func(*[i.value for i in row])
