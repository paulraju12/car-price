car['kms_driven']=car['kms_driven'].str.split(" ").str.get(0).str.replace(',','')
# car=car[car['kms_driven'].str.isnumeric()]
# car['kms_driven']=car['kms_driven'].astype(int)