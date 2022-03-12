from data_loader import MAMeDataset


mema_dataset = MAMeDataset(3,4, 'test')

x, y = mema_dataset[1]
print(x)
print(y)